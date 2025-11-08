import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import NavigationHeader from '../../components/ui/NavigationHeader';
import MapContainer from './components/MapContainer';
import FilterControls from './components/FilterControls';
import PropertyListPanel from './components/PropertyListPanel';
import PropertyDetailModal from './components/PropertyDetailModal';
import Icon from '../../components/AppIcon';
import Button from '../../components/ui/Button';

const defaultCities = [
  { name: "Bratislava", coords: { lat: 48.1486, lng: 17.1077 } },
  { name: "Košice", coords: { lat: 48.7164, lng: 21.2611 } },
  { name: "Žilina", coords: { lat: 49.2231, lng: 18.7394 } },
  { name: "Trnava", coords: { lat: 48.3774, lng: 17.5872 } },
  { name: "Nitra", coords: { lat: 48.3069, lng: 18.0840 } },
  { name: "Banská Bystrica", coords: { lat: 48.7363, lng: 19.1462 } },
  { name: "Prešov", coords: { lat: 48.9976, lng: 21.2339 } },
];

const MapView = () => {
  const [rawProperties, setProperties] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);
  const [sourceMeta, setSourceMeta] = useState(null);
  const [selectedProperty, setSelectedProperty] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showMobileFilters, setShowMobileFilters] = useState(false);
  const [activeView, setActiveView] = useState('map'); // 'map' or 'list' for mobile
  const [mapFocus, setMapFocus] = useState(null);
  const mapSectionRef = useRef(null);
  const [filters, setFilters] = useState({
    propertyType: 'all',
    region: 'all',
    priceRange: { min: null, max: null },
    minCapRate: null,
    minROI: null
  });
  const properties = Array.isArray(rawProperties) ? rawProperties : [];

  const loadProperties = useCallback(async () => {
    setIsLoading(true);
    setLoadError(null);
    try {
      const response = await fetch("/api/listings?limit=400");
      if (!response.ok) {
        throw new Error(`Server vrátil stav ${response.status}`);
      }
      const payload = await response.json();
      const next =
        Array.isArray(payload?.data)
          ? payload.data
          : Array.isArray(payload)
            ? payload
            : [];
      setProperties(next);
      setSourceMeta(payload?.meta || null);
    } catch (err) {
      setLoadError(err?.message || "Nepodarilo sa načítať dáta");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProperties();
  }, [loadProperties]);

  useEffect(() => {
    if (!selectedProperty) return;
    const stillExists = Array.isArray(rawProperties)
      ? rawProperties.some((property) => property?.id === selectedProperty?.id)
      : false;
    if (!stillExists) {
      setSelectedProperty(null);
      setShowDetailModal(false);
    }
  }, [properties, selectedProperty]);

  useEffect(() => {
    if (selectedProperty) {
      setMapFocus(null);
    }
  }, [selectedProperty]);

  const handlePropertySelect = (property) => {
    setSelectedProperty(property);
  };

  const handlePropertyDetail = (property) => {
    setSelectedProperty(property);
    setShowDetailModal(true);
  };

  const handleFiltersChange = (newFilters) => {
    setFilters(newFilters);
  };

  const handleClearFilters = () => {
    setFilters({
      propertyType: 'all',
      region: 'all',
      priceRange: { min: null, max: null },
      minCapRate: null,
      minROI: null
    });
  };

  const toggleMobileFilters = () => {
    setShowMobileFilters(!showMobileFilters);
  };

  const parquetLabel = sourceMeta?.parquet_path
    ? sourceMeta.parquet_path.split(/[/\\]/).slice(-3).join("/")
    : null;
  const totalListings = sourceMeta?.total ?? sourceMeta?.total_rows ?? properties.length;

  const cityTiles = useMemo(() => {
    if (!properties?.length) {
      return defaultCities.map((city) => ({
        name: city.name,
        count: 0,
        focus: { center: city.coords, zoom: 11 },
      }));
    }

    const buckets = defaultCities.reduce((acc, city) => {
      acc[city.name.toLowerCase()] = {
        name: city.name,
        count: 0,
        bounds: null,
        coords: city.coords,
      };
      return acc;
    }, {});

    properties.forEach((property) => {
      const cityName =
        property?.city ||
        property?.address_town ||
        property?.addressTown ||
        property?.address_town_norm;
      const normalized = cityName?.trim();
      if (!normalized) return;
      const key = normalized.toLowerCase();
      if (!buckets[key]) return;
      const bucket = buckets[key];
      bucket.count += 1;
      if (property?.lat != null && property?.lng != null) {
        const lat = Number(property.lat);
        const lng = Number(property.lng);
        if (Number.isFinite(lat) && Number.isFinite(lng)) {
          if (!bucket.bounds) {
            bucket.bounds = { north: lat, south: lat, east: lng, west: lng };
          } else {
            bucket.bounds = {
              north: Math.max(bucket.bounds.north, lat),
              south: Math.min(bucket.bounds.south, lat),
              east: Math.max(bucket.bounds.east, lng),
              west: Math.min(bucket.bounds.west, lng),
            };
          }
        }
      }
    });

    return defaultCities.map((city) => {
      const bucket = buckets[city.name.toLowerCase()];
      if (!bucket) {
        return {
          name: city.name,
          count: 0,
          focus: { center: city.coords, zoom: 11 },
        };
      }

      const focus = bucket.bounds
        ? {
            center: {
              lat: (bucket.bounds.north + bucket.bounds.south) / 2,
              lng: (bucket.bounds.east + bucket.bounds.west) / 2,
            },
            zoom: 12,
          }
        : { center: city.coords, zoom: 11 };

      return {
        name: bucket.name,
        count: bucket.count,
        focus,
      };
    });
  }, [properties]);

  return (
    <div className="min-h-screen bg-slate-100 text-black">
      <NavigationHeader />

      <main className="mx-auto max-w-6xl px-4 pb-10 sm:px-6 lg:px-8 pt-10">
        <div className="rounded-3xl border border-white/10 bg-white/5 shadow-[0_20px_70px_rgba(2,6,23,0.45)] backdrop-blur-3xl p-6">
          <div className="mb-6 text-center">
            <p className="text-xs uppercase tracking-[0.5em] text-black">Investičný poradca</p>
            <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
              {cityTiles.map((city) => (
                <button
                  key={city.name}
                  className="rounded-2xl border border-white/20 bg-white/10 px-4 py-3 text-center shadow-[0_15px_35px_rgba(2,6,23,0.25)] hover:border-white/40 hover:bg-white/20 transition focus:outline-none focus:ring-2 focus:ring-white/50"
                  onClick={() => {
                    setMapFocus(city.focus);
                    if (mapSectionRef.current) {
                      mapSectionRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
                    }
                  }}
                >
                  <p className="text-[11px] font-semibold tracking-wide text-black">{city.name}</p>
                  <p className="text-xs text-black/70">
                    {city.count !== undefined ? `${city.count} ponúk` : "—"}
                  </p>
                </button>
              ))}
            </div>
          </div>
          <FilterControls
            filters={filters}
            onFiltersChange={handleFiltersChange}
            onClearFilters={handleClearFilters}
            isVisible={showMobileFilters}
            onToggle={toggleMobileFilters}
          />
          {sourceMeta && (
            <p className="mt-3 text-xs text-black">
              Zdroj dát: {parquetLabel || sourceMeta.parquet_path} • {totalListings} záznamov
            </p>
          )}
        </div>

        <div className="mt-8 lg:hidden mb-4">
          <div className="flex bg-slate-100 rounded-2xl p-1 shadow-inner">
            {["map", "list"].map((view) => (
              <button
                key={view}
                onClick={() => setActiveView(view)}
                className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition ${
                  activeView === view ? "bg-white shadow text-black" : "text-black"
                }`}
              >
                <Icon name={view === "map" ? "Map" : "List"} size={16} />
                {view === "map" ? "Mapa" : "Zoznam"}
              </button>
            ))}
          </div>
        </div>

        <div ref={mapSectionRef} className="mt-6 grid gap-6 lg:grid-cols-[420px,1fr]">
          <div className="hidden lg:block">
            <PropertyListPanel
              properties={properties}
              selectedProperty={selectedProperty}
              onPropertySelect={handlePropertySelect}
              filters={filters}
              isLoading={isLoading}
              error={loadError}
              onRetry={loadProperties}
            />
          </div>

          <div className="min-h-[520px]">
            <MapContainer
              properties={properties}
              selectedProperty={selectedProperty}
              onPropertySelect={handlePropertySelect}
              focusQuery={mapFocus}
              filters={filters}
              isLoading={isLoading}
              error={loadError}
              onRetry={loadProperties}
            />
          </div>
        </div>

        <div className="lg:hidden mt-6">
          {activeView === "list" && (
            <PropertyListPanel
              properties={properties}
              selectedProperty={selectedProperty}
              onPropertySelect={handlePropertySelect}
              filters={filters}
              isLoading={isLoading}
              error={loadError}
              onRetry={loadProperties}
            />
          )}
        </div>
      </main>

      <PropertyDetailModal
        property={selectedProperty}
        isOpen={showDetailModal}
        onClose={() => setShowDetailModal(false)}
      />
    </div>
  );
};

export default MapView;
