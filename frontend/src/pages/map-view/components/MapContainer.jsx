import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Icon from "../../../components/AppIcon";

const SLOVAKIA_CENTER = { lat: 48.669, lng: 19.699 };
const LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
const LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";

let leafletPromise = null;
const loadLeaflet = () => {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Leaflet can only load in the browser"));
  }
  if (window.L) return Promise.resolve(window.L);
  if (leafletPromise) return leafletPromise;

  leafletPromise = new Promise((resolve, reject) => {
    const existingScript = document.getElementById("leaflet-script");

    const onLoad = () => resolve(window.L);
    const onError = () => reject(new Error("Failed to load Leaflet assets"));

    if (!existingScript) {
      const script = document.createElement("script");
      script.src = LEAFLET_JS;
      script.id = "leaflet-script";
      script.async = true;
      script.onload = onLoad;
      script.onerror = onError;
      document.body.appendChild(script);
    } else if (window.L) {
      resolve(window.L);
    } else {
      existingScript.addEventListener("load", onLoad, { once: true });
      existingScript.addEventListener("error", onError, { once: true });
    }

    if (!document.getElementById("leaflet-css")) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = LEAFLET_CSS;
      link.id = "leaflet-css";
      document.head.appendChild(link);
    }
  });

  return leafletPromise;
};

const MapContainer = ({
  properties = [],
  selectedProperty,
  onPropertySelect,
  filters,
  isLoading,
  error,
  onRetry,
  focusQuery,
}) => {
  const outerShellRef = useRef(null);
  const mapNodeRef = useRef(null);
  const mapRef = useRef(null);
  const markersLayerRef = useRef(null);
  const leafletRef = useRef(null);
  const [visibleCount, setVisibleCount] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const filteredProperties = useMemo(
    () =>
      (properties ?? []).filter((property) => {
        if (filters?.propertyType && filters.propertyType !== "all" && property?.type !== filters.propertyType) {
          return false;
        }
        if (filters?.priceRange?.min && property?.price < filters.priceRange.min) {
          return false;
        }
        if (filters?.priceRange?.max && property?.price > filters.priceRange.max) {
          return false;
        }
        if (filters?.region && filters.region !== "all" && property?.region !== filters.region) {
          return false;
        }
        if (filters?.minCapRate && property?.capRate < filters.minCapRate) {
          return false;
        }
        if (filters?.minROI && property?.roi < filters.minROI) {
          return false;
        }
        return true;
      }),
    [properties, filters]
  );

  const ensureMap = useCallback(() => {
    if (!leafletRef.current || !mapNodeRef.current || mapRef.current) return;
    const L = leafletRef.current;
    const map = L.map(mapNodeRef.current, {
      center: [SLOVAKIA_CENTER.lat, SLOVAKIA_CENTER.lng],
      zoom: 7,
      scrollWheelZoom: true,
      zoomControl: false,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    mapRef.current = map;
    markersLayerRef.current = L.layerGroup().addTo(map);
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadLeaflet()
      .then((L) => {
        if (cancelled) return;
        leafletRef.current = L;
        ensureMap();
      })
      .catch((err) => {
        console.error("Leaflet failed to load", err);
      });

    return () => {
      cancelled = true;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
        markersLayerRef.current = null;
      }
    };
  }, [ensureMap]);

  const flyToLocation = useCallback((lat, lng, zoom = 11) => {
    const map = mapRef.current;
    if (!map) return;
    const latNum = Number(lat);
    const lngNum = Number(lng);
    if (!Number.isFinite(latNum) || !Number.isFinite(lngNum)) return;
    map.flyTo([latNum, lngNum], zoom, { duration: 0.6 });
  }, []);

  const focusOnBounds = useCallback((bounds, zoom) => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L || !bounds) return;
    const { north, south, east, west } = bounds;
    const parsed = [north, south, east, west].map((value) => Number(value));
    if (parsed.some((value) => !Number.isFinite(value))) return;
    const latLngBounds = L.latLngBounds(
      [parsed[1], parsed[3]],
      [parsed[0], parsed[2]]
    );
    map.flyToBounds(latLngBounds, { padding: [24, 24], duration: 0.6 });
    if (zoom) {
      map.once("moveend", () => map.setZoom(zoom));
    }
  }, []);

  const getMarkerColor = (property) => {
    if (property?.investmentScore >= 8) return "#38A169";
    if (property?.investmentScore >= 6) return "#D69E2E";
    return "#E53E3E";
  };

  const renderMarkers = useCallback(() => {
    const map = mapRef.current;
    const layer = markersLayerRef.current;
    const L = leafletRef.current;
    if (!map || !layer || !L) return;

    layer.clearLayers();
    let count = 0;

    filteredProperties.forEach((property) => {
      const lat = Number(property?.lat);
      const lng = Number(property?.lng);
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
        return;
      }

      count += 1;
      const marker = L.circleMarker([lat, lng], {
        radius: selectedProperty?.id === property?.id ? 9 : 7,
        color: getMarkerColor(property),
        weight: 2,
        fillOpacity: 0.85,
      });

      marker.on("click", () => {
        flyToLocation(lat, lng, Math.max(map.getZoom(), 13));
        onPropertySelect(property);
      });

      marker.bindTooltip(
        `${property?.title || "Neznáma ponuka"}<br/>€${property?.price?.toLocaleString("sk-SK") || "N/A"}`,
        { direction: "top" }
      );

      marker.addTo(layer);
    });

    setVisibleCount(count);
  }, [filteredProperties, selectedProperty, flyToLocation, onPropertySelect]);

  useEffect(() => {
    renderMarkers();
  }, [renderMarkers]);

  useEffect(() => {
    if (typeof document === "undefined") return undefined;
    const handler = () => {
      setIsFullscreen(document.fullscreenElement === outerShellRef.current);
    };
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (typeof document === "undefined" || !outerShellRef.current) return;
    if (document.fullscreenElement === outerShellRef.current) {
      document.exitFullscreen?.();
    } else {
      outerShellRef.current.requestFullscreen?.().catch(() => {});
    }
  }, []);

  useEffect(() => {
    if (!selectedProperty) return;
    flyToLocation(selectedProperty.lat, selectedProperty.lng, Math.max(mapRef.current?.getZoom() || 7, 13));
  }, [selectedProperty, flyToLocation]);

  useEffect(() => {
    if (!focusQuery) return;
    if (focusQuery.center) {
      flyToLocation(focusQuery.center.lat, focusQuery.center.lng, focusQuery.zoom || 11);
    } else if (focusQuery.bounds) {
      focusOnBounds(focusQuery.bounds, focusQuery.zoom);
    }
  }, [focusQuery, flyToLocation, focusOnBounds]);

  const stateMessage = (() => {
    if (isLoading) {
      return { title: "Načítavam mapu…", description: "Získavam dáta z posledného parquet exportu." };
    }
    if (error) {
      return { title: "Nepodarilo sa načítať dáta", description: error };
    }
    if (!visibleCount) {
      return { title: "Žiadne ponuky pre zvolené filtre", description: "Skúste upraviť parametre." };
    }
    return null;
  })();

  return (
    <div
      ref={outerShellRef}
      className="relative w-full min-h-[520px] h-full rounded-3xl border border-white/80 bg-gradient-to-br from-slate-900 to-slate-800 overflow-hidden shadow-[0_35px_120px_rgba(15,23,42,0.35)]"
    >
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.08),_transparent_55%)]" />
      </div>

      <div className="absolute top-4 left-4 right-4 z-10 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="rounded-2xl bg-white/90 px-4 py-2 shadow-lg backdrop-blur">
          <p className="text-xs uppercase tracking-[0.4em] text-slate-400">Na mape</p>
          <p className="text-xl font-semibold text-slate-900">{visibleCount} aktívnych listingov</p>
        </div>
      </div>

      <div ref={mapNodeRef} className="w-full h-full min-h-[520px]" />

      {stateMessage && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/90 backdrop-blur">
          <Icon
            name={error ? "AlertCircle" : "Loader2"}
            size={32}
            className={`mb-3 text-muted-foreground ${error ? "" : "animate-spin"}`}
          />
          <p className="text-lg font-semibold text-foreground mb-1 text-center px-6">{stateMessage.title}</p>
          <p className="text-sm text-muted-foreground text-center px-8">{stateMessage.description}</p>
          {error && onRetry && (
            <button
              className="mt-4 inline-flex items-center rounded-full bg-primary px-4 py-2 text-sm font-semibold text-white shadow hover:bg-primary/90 transition"
              onClick={onRetry}
            >
              Skúsiť znova
            </button>
          )}
        </div>
      )}

      <div className="absolute bottom-4 left-4 rounded-2xl bg-white/90 backdrop-blur shadow-lg border border-white/70 p-4">
        <p className="text-xs uppercase tracking-[0.35em] text-slate-400 mb-2">Investičné skóre</p>
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-success" />
            <span className="text-xs text-muted-foreground">Vysoké (8-10)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-warning" />
            <span className="text-xs text-muted-foreground">Stredné (6-7)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-destructive" />
            <span className="text-xs text-muted-foreground">Nízke (0-5)</span>
          </div>
        </div>
      </div>
      <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-2">
        <button
          className="h-10 w-10 rounded-full bg-white text-slate-700 shadow hover:text-blue-600 transition flex items-center justify-center border border-slate-200/80"
          onClick={() => mapRef.current?.zoomIn()}
          aria-label="Priblížiť"
        >
          <Icon name="ZoomIn" size={16} />
        </button>
        <button
          className="h-10 w-10 rounded-full bg-white text-slate-700 shadow hover:text-blue-600 transition flex items-center justify-center border border-slate-200/80"
          onClick={() => mapRef.current?.zoomOut()}
          aria-label="Oddialiť"
        >
          <Icon name="ZoomOut" size={16} />
        </button>
        <button
          className="h-10 w-10 rounded-full bg-white text-slate-700 shadow hover:text-blue-600 transition flex items-center justify-center border border-slate-200/80"
          onClick={toggleFullscreen}
          aria-label="Prepínať celú obrazovku"
        >
          <Icon name={isFullscreen ? "Minimize" : "Maximize"} size={15} />
        </button>
      </div>
    </div>
  );
};

export default MapContainer;
