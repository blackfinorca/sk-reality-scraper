import React, { useState, useEffect, useCallback } from 'react';
import NavigationHeader from '../../components/ui/NavigationHeader';
import MapContainer from './components/MapContainer';
import FilterControls from './components/FilterControls';
import PropertyListPanel from './components/PropertyListPanel';
import PropertyDetailModal from './components/PropertyDetailModal';
import Icon from '../../components/AppIcon';
import Button from '../../components/ui/Button';

const MapView = () => {
  const [properties, setProperties] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);
  const [sourceMeta, setSourceMeta] = useState(null);
  const [selectedProperty, setSelectedProperty] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showMobileFilters, setShowMobileFilters] = useState(false);
  const [activeView, setActiveView] = useState('map'); // 'map' or 'list' for mobile
  const [filters, setFilters] = useState({
    propertyType: 'all',
    region: 'all',
    priceRange: { min: null, max: null },
    minCapRate: null,
    minROI: null
  });

  const loadProperties = useCallback(async () => {
    setIsLoading(true);
    setLoadError(null);
    try {
      const response = await fetch("/api/listings?limit=400");
      if (!response.ok) {
        throw new Error(`Server vrátil stav ${response.status}`);
      }
      const payload = await response.json();
      setProperties(Array.isArray(payload?.data) ? payload.data : payload);
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
    const stillExists = properties.some((property) => property?.id === selectedProperty?.id);
    if (!stillExists) {
      setSelectedProperty(null);
      setShowDetailModal(false);
    }
  }, [properties, selectedProperty]);

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

  return (
    <div className="min-h-screen bg-slate-100 text-black">
      <NavigationHeader />

      <main className="mx-auto max-w-6xl px-4 pb-10 sm:px-6 lg:px-8 pt-10">
        <div className="rounded-3xl border border-white/10 bg-white/5 shadow-[0_20px_70px_rgba(2,6,23,0.45)] backdrop-blur-3xl p-6">
          <div className="mb-6 text-center">
            <p className="text-xs uppercase tracking-[0.5em] text-black">Investičný poradca</p>
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

        <div className="mt-6 grid gap-6 lg:grid-cols-[420px,1fr]">
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
