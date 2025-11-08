import React, { useState } from "react";
import Icon from "../../../components/AppIcon";
import Image from "../../../components/AppImage";
import Button from "../../../components/ui/Button";

export default function PropertyListPanel({
  properties = [],
  selectedProperty,
  onPropertySelect,
  filters,
  isLoading,
  error,
  onRetry,
}) {
  const [saved, setSaved] = useState(new Set());

  const filtered = properties.filter((property) => {
    if (filters.propertyType !== "all" && property.type !== filters.propertyType) return false;
    if (filters.region !== "all" && property.region !== filters.region) return false;
    if (filters.priceRange.min && property.price < filters.priceRange.min) return false;
    if (filters.priceRange.max && property.price > filters.priceRange.max) return false;
    if (filters.minCapRate && property.capRate < filters.minCapRate) return false;
    if (filters.minROI && property.roi < filters.minROI) return false;
    return true;
  });

  const toggleSave = (id, event) => {
    event?.stopPropagation();
    const copy = new Set(saved);
    if (copy.has(id)) copy.delete(id);
    else copy.add(id);
    setSaved(copy);
  };

  const typeLabel = (type) =>
    (
      {
        apartment: "Byt",
        house: "Dom",
        commercial: "Komerčné",
        land: "Pozemok",
      }[type] || type
    );

  const body = (() => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-center space-y-3">
          <Icon name="Loader2" size={32} className="text-blue-500 animate-spin" />
          <p className="text-sm text-slate-500">Načítavam investičné príležitosti…</p>
        </div>
      );
    }
    if (error) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-center space-y-3">
          <Icon name="AlertTriangle" size={32} className="text-red-500" />
          <p className="text-sm text-slate-500">{error}</p>
          {onRetry && (
            <Button variant="outline" iconName="RefreshCcw" onClick={onRetry}>
              Skúsiť znova
            </Button>
          )}
        </div>
      );
    }
    if (!filtered.length) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-center space-y-3">
          <Icon name="Search" size={32} className="text-slate-400" />
          <p className="text-sm text-slate-500">Žiadne ponuky nespĺňajú aktuálne filtre.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {filtered.map((property) => {
          const active = selectedProperty?.id === property.id;
          return (
            <button
              key={property.id}
              onClick={() => onPropertySelect(property)}
              className={`w-full text-left rounded-3xl border backdrop-blur bg-white/95 transition shadow-sm hover:shadow-lg ${
                active ? "border-blue-300 shadow-lg" : "border-white"
              }`}
            >
            <div className="relative overflow-hidden rounded-2xl">
              {property.hasScrapedPhotos ? (
                <Image
                  src={property.images?.[0]}
                  alt={property.imageAlts?.[0]}
                  className="h-48 w-full object-cover"
                />
              ) : (
                <div className="h-48 w-full bg-slate-200 animate-pulse" />
              )}
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900/40 to-transparent" />
                <div className="absolute top-3 left-3 flex items-center gap-2">
                  <span className="rounded-full bg-white/20 px-3 py-1 text-xs font-semibold text-white backdrop-blur">
                    Skóre {property.investmentScore}/10
                  </span>
                  <span className="rounded-full bg-white/20 px-3 py-1 text-xs font-semibold text-white backdrop-blur">
                    {typeLabel(property.type)}
                  </span>
                </div>
                <button
                  onClick={(event) => toggleSave(property.id, event)}
                  className="absolute top-3 right-3 rounded-full bg-white/90 p-2 shadow"
                >
                  <Icon
                    name="Heart"
                    size={16}
                    className={saved.has(property.id) ? "text-red-500 fill-red-500" : "text-slate-400"}
                  />
                </button>
              </div>
              <div className="p-4 space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-sm uppercase tracking-[0.35em] text-slate-400">
                      {property.city}
                    </p>
                    <h3 className="mt-1 text-lg font-semibold text-slate-900 line-clamp-2">
                      {property.title}
                    </h3>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-blue-600">
                      {property.price ? `€${property.price.toLocaleString("sk-SK")}` : "N/A"}
                    </p>
                    <p className="text-xs text-slate-500">
                      {property.pricePerSqm ? `€${property.pricePerSqm}/m²` : "–"}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-slate-500">
                  <span className="inline-flex items-center gap-1">
                    <Icon name="Bed" size={14} />
                    {property.bedrooms || 0} izby
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <Icon name="Bath" size={14} />
                    {property.bathrooms || 1} kúpeľne
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <Icon name="Square" size={14} />
                    {property.area || 0} m²
                  </span>
                </div>
                <div className="rounded-2xl bg-slate-100 px-3 py-2 text-xs text-slate-500 flex justify-between">
                  <span>Cap rate {property.capRate || "–"} %</span>
                  <span>ROI {property.roi || "–"} %</span>
                  <span>Cashflow €{property.monthlyRent || "–"}</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    );
  })();

  return (
    <div className="h-full rounded-3xl border border-white/70 bg-white/95 p-4 shadow-[0_20px_60px_rgba(15,23,42,0.1)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Investičný shortlist</p>
          <h2 className="text-lg font-semibold text-slate-900">Top nehnuteľnosti</h2>
        </div>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-500">
          {filtered.length} výsledkov
        </span>
      </div>
      <div className="h-[calc(100vh-240px)] overflow-y-auto pr-2">{body}</div>
    </div>
  );
}
