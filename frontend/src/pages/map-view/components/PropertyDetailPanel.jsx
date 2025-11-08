import React from "react";
import Icon from "../../../components/AppIcon";
import Image from "../../../components/AppImage";

const formatNumber = (value) =>
  typeof value === "number" && !Number.isNaN(value) ? value.toLocaleString("sk-SK") : "—";

const PropertyDetailPanel = ({ property, onBack }) => {
  if (!property) return null;

  const photos = property.images?.length ? property.images : [];
  const mainPhoto = photos[0];
  const otherPhotos = photos.slice(1, 3);
  const sourceUrl = property.primary_source_url || property.url;

  const cleanSummary = (text) => {
    if (!text) return null;
    const pattern = /summary_short_sk:\s*"([^"]+)"/i;
    const match = text.match(pattern);
    if (match) return match[1].trim();
    return text
      .replace(/INPUT DESCRIPTION:[\s\S]*?OUTPUT:\s*/i, "")
      .replace(/summary_short_sk:\s*/i, "")
      .replace(/^"+|"+$/g, "")
      .trim();
  };

  const summaryText = cleanSummary(property.summary);

  return (
    <div className="flex h-full flex-col rounded-3xl border border-white/70 bg-white/95 shadow-[0_20px_60px_rgba(15,23,42,0.1)]">
      <div className="flex items-center justify-between px-5 pt-4">
        <div>
          <p className="text-[0.7rem] uppercase tracking-[0.35em] text-slate-500">Detail ponuky</p>
          <h2 className="text-xl font-semibold text-slate-900 line-clamp-2 mt-1">{property.title}</h2>
        </div>
        {onBack && (
          <button
            type="button"
            onClick={onBack}
            className="h-9 w-9 rounded-full border border-slate-200 text-slate-500 hover:text-slate-900 flex items-center justify-center shadow-sm"
            aria-label="Naspäť na zoznam"
          >
            <Icon name="ChevronLeft" size={18} />
          </button>
        )}
      </div>

      <div className="px-5 py-4 space-y-4 overflow-y-auto">
        {photos.length > 0 ? (
          <div className="space-y-3">
            {mainPhoto && (
              <Image
                src={mainPhoto}
                alt={property.title}
                className="w-full h-56 rounded-2xl object-cover"
                loading="lazy"
              />
            )}
            {otherPhotos.length > 0 && (
              <div className="grid grid-cols-2 gap-3">
                {otherPhotos.map((photo, index) => (
                  <Image
                    key={photo + index}
                    src={photo}
                    alt={`${property.title} ${index + 2}`}
                    className="h-40 w-full rounded-xl object-cover"
                    loading="lazy"
                  />
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            <div className="col-span-2 h-56 rounded-2xl bg-slate-200 animate-pulse" />
            <div className="h-40 rounded-xl bg-slate-200 animate-pulse" />
            <div className="h-40 rounded-xl bg-slate-200 animate-pulse" />
          </div>
        )}

        <div className="grid grid-cols-2 gap-3 rounded-2xl border border-slate-100 bg-slate-50 p-4">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Cena</p>
            <p className="text-lg font-semibold text-slate-900">
              €{formatNumber(property.price)}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Výmera</p>
            <p className="text-lg font-semibold text-slate-900">
              {property.area ? `${formatNumber(property.area)} m²` : "—"}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Izby</p>
            <p className="text-lg font-semibold text-slate-900">{property.bedrooms || "—"}</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Kúpeľne</p>
            <p className="text-lg font-semibold text-slate-900">{property.bathrooms || "—"}</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-slate-400">€/m²</p>
            <p className="text-lg font-semibold text-slate-900">
              €{property.pricePerSqm ? Math.round(property.pricePerSqm).toLocaleString("sk-SK") : "—"}
            </p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Cap rate</p>
            <p className="text-lg font-semibold text-slate-900">{property.capRate || "—"} %</p>
          </div>
        </div>

        {sourceUrl && (
          <a
            href={sourceUrl}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-3 rounded-2xl border border-blue-100 bg-blue-50 px-4 py-3 text-sm font-semibold text-blue-700 hover:bg-blue-100 transition"
          >
            <Icon name="ExternalLink" size={16} />
            Zdroj: {property.portal || "neznámy"}
          </a>
        )}

        {summaryText && (
          <div className="rounded-2xl border border-slate-100 bg-white px-4 py-3">
            <p className="text-xs uppercase tracking-[0.35em] text-slate-400 mb-2">AI zhrnutie</p>
            <p className="text-sm text-slate-700 leading-relaxed">{summaryText}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PropertyDetailPanel;
