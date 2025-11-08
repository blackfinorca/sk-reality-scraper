import React, { useState, useEffect } from 'react';
import Icon from '../../../components/AppIcon';

const MapContainer = ({
  properties = [],
  selectedProperty,
  onPropertySelect,
  filters,
  isLoading,
  error,
  onRetry,
}) => {
  const [mapCenter] = useState({ lat: 48.6690, lng: 19.6990 }); // Slovakia center
  const [zoom] = useState(8);

  // Filter properties based on current filters
  const filteredProperties = properties?.filter(property => {
    if (filters?.propertyType && filters?.propertyType !== 'all' && property?.type !== filters?.propertyType) {
      return false;
    }
    if (filters?.priceRange?.min && property?.price < filters?.priceRange?.min) {
      return false;
    }
    if (filters?.priceRange?.max && property?.price > filters?.priceRange?.max) {
      return false;
    }
    if (filters?.region && filters?.region !== 'all' && property?.region !== filters?.region) {
      return false;
    }
    if (filters?.minCapRate && property?.capRate < filters?.minCapRate) {
      return false;
    }
    if (filters?.minROI && property?.roi < filters?.minROI) {
      return false;
    }
    return true;
  });

  const getMarkerColor = (property) => {
    if (property?.investmentScore >= 8) return '#38A169'; // green
    if (property?.investmentScore >= 6) return '#D69E2E'; // yellow
    return '#E53E3E'; // red
  };

  const handleMarkerClick = (property) => {
    onPropertySelect(property);
  };

  const stateMessage = (() => {
    if (isLoading) {
      return { title: "Načítavam mapu…", description: "Získavam dáta z posledného parquet exportu." };
    }
    if (error) {
      return { title: "Nepodarilo sa načítať dáta", description: error };
    }
    if (!filteredProperties?.length) {
      return { title: "Žiadne ponuky pre zvolené filtre", description: "Skúste upraviť parametre." };
    }
    return null;
  })();

  return (
    <div className="relative w-full h-full rounded-3xl border border-white/80 bg-gradient-to-br from-slate-900 to-slate-800 overflow-hidden shadow-[0_35px_120px_rgba(15,23,42,0.35)]">
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.08),_transparent_55%)]" />
      </div>

      <div className="absolute top-4 left-4 right-4 z-10 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="rounded-2xl bg-white/90 px-4 py-2 shadow-lg backdrop-blur">
          <p className="text-xs uppercase tracking-[0.4em] text-slate-400">Na mape</p>
          <p className="text-xl font-semibold text-slate-900">{filteredProperties.length} aktívnych listingov</p>
        </div>
        <div className="flex items-center gap-2">
          {["ZoomIn", "ZoomOut"].map((action) => (
            <button
              key={action}
              className="p-2 rounded-2xl bg-white/80 text-slate-600 shadow hover:text-blue-600 transition"
            >
              <Icon name={action} size={18} />
            </button>
          ))}
        </div>
      </div>

      <iframe
        width="100%"
        height="100%"
        loading="lazy"
        title="Mapa nehnuteľností na Slovensku"
        referrerPolicy="no-referrer-when-downgrade"
        src={`https://www.google.com/maps?q=${mapCenter?.lat},${mapCenter?.lng}&z=${zoom}&output=embed`}
        className="w-full h-full border-0"
      />
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
      {/* Property Markers Overlay */}
      <div className="absolute inset-0 pointer-events-none">
        {filteredProperties?.map((property, index) => (
          <div
            key={property?.id}
            className="absolute pointer-events-auto cursor-pointer transform -translate-x-1/2 -translate-y-1/2"
            style={{
              left: `${20 + (index % 8) * 10}%`,
              top: `${30 + Math.floor(index / 8) * 15}%`
            }}
            onClick={() => handleMarkerClick(property)}
          >
            <div
              className={`relative bg-card border-2 rounded-lg px-2 py-1 shadow-lg transition-all duration-200 hover:scale-110 ${
                selectedProperty?.id === property?.id
                  ? 'border-primary scale-110' :'border-border hover:border-primary'
              }`}
            >
              <div className="flex items-center space-x-1">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getMarkerColor(property) }}
                />
                <span className="text-xs font-medium text-foreground whitespace-nowrap">
                  €{property?.price?.toLocaleString('sk-SK')}
                </span>
              </div>
              <div
                className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-full w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent"
                style={{ borderTopColor: selectedProperty?.id === property?.id ? 'var(--color-primary)' : 'var(--color-border)' }}
              />
            </div>
          </div>
        ))}
      </div>
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
    </div>
  );
};

export default MapContainer;
