import React from "react";
import Button from "../../../components/ui/Button";

const propertyTypeOptions = [
  { value: "apartment", label: "Flat" },
  { value: "house", label: "House" },
  { value: "commercial", label: "Commercial" },
];

const priceSlider = {
  min: 50000,
  max: 1000000,
  step: 10000,
};

const FilterControls = ({ filters, onFiltersChange, onClearFilters, isVisible, onToggle }) => {
  const handlePropertyType = (value) => {
    const nextValue = filters.propertyType === value ? "all" : value;
    onFiltersChange({ ...filters, propertyType: nextValue });
  };

  const currentMax =
    filters.priceRange?.max != null ? filters.priceRange.max : priceSlider.max;

  const handleSliderChange = (value) => {
    onFiltersChange({
      ...filters,
      priceRange: {
        min: priceSlider.min,
        max: Number(value),
      },
    });
  };

  const renderTypeIcon = (type) => {
    const common = {
      width: 28,
      height: 28,
      viewBox: "0 0 32 32",
      fill: "none",
      "aria-hidden": "true",
      stroke: "currentColor",
      strokeWidth: 1.6,
      strokeLinecap: "round",
      strokeLinejoin: "round",
    };

    switch (type) {
      case "apartment":
        return (
          <svg {...common}>
            <rect x="9" y="8" width="14" height="18" rx="2" />
            <path d="M12 12v4M20 12v4M12 19v4M20 19v4" />
          </svg>
        );
      case "house":
        return (
          <svg {...common}>
            <path d="M5 15.5 16 7l11 8.5v10a1.5 1.5 0 0 1-1.5 1.5H20v-6h-8v6h-5.5A1.5 1.5 0 0 1 5 25.5v-10Z" />
            <path d="M2 17 16 6l14 11" />
          </svg>
        );
      case "commercial":
      default:
        return (
          <svg {...common}>
            <rect x="8" y="7" width="16" height="19" rx="2" />
            <path d="M11 12h10M11 16h10M11 20h7" />
          </svg>
        );
    }
  };

  const renderPropertyButtons = (className = "") => (
    <div className={`grid grid-cols-3 gap-2 w-full ${className}`}>
      {propertyTypeOptions.map((option) => {
        const active = filters.propertyType === option.value;
        return (
          <button
            key={option.value}
            onClick={() => handlePropertyType(option.value)}
            aria-pressed={active}
            aria-label={option.label}
            className={`h-16 w-16 mx-auto rounded-full border flex items-center justify-center transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
              active
                ? "border-slate-900 bg-white text-black shadow-lg"
                : "border-slate-300 bg-white text-black hover:bg-slate-50"
            }`}
          >
            {renderTypeIcon(option.value)}
            <span className="sr-only">{option.label}</span>
          </button>
        );
      })}
    </div>
  );

  const renderPriceBlock = () => (
    <div className="rounded-2xl border border-slate-300 bg-white px-4 py-3 shadow-inner">
      <div>
        <p className="text-[10px] uppercase tracking-[0.5em] text-black">Cenové pásmo</p>
        <p className="text-lg font-semibold text-black">do €{currentMax.toLocaleString("sk-SK")}</p>
      </div>
      <div className="mt-3">
        <input
          type="range"
          min={priceSlider.min}
          max={priceSlider.max}
          step={priceSlider.step}
          value={currentMax}
          onChange={(e) => handleSliderChange(e.target.value)}
          className="w-full accent-blue-300"
        />
        <div className="mt-2 flex justify-between text-xs text-black">
          <span>€{priceSlider.min.toLocaleString("sk-SK")}</span>
          <span>€{priceSlider.max.toLocaleString("sk-SK")}</span>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <div className="hidden lg:block">
        <div className="rounded-3xl border border-slate-200 bg-white px-5 py-3 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
          <div className="mt-1 flex flex-col gap-3 lg:flex-row lg:items-center">
            <div className="flex-1">{renderPropertyButtons("justify-center lg:justify-start")}</div>
            <div className="w-full lg:max-w-md shrink-0">{renderPriceBlock()}</div>
          </div>
        </div>
      </div>

      <div className="lg:hidden">
        <Button
          variant="outline"
          onClick={onToggle}
          iconName="Filter"
          iconPosition="left"
          className="w-full rounded-2xl border-slate-300 bg-white text-black"
        >
          Filtrovanie
        </Button>

        {isVisible && (
          <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm">
            <div className="fixed inset-y-0 right-0 w-full max-w-sm bg-slate-950 text-white rounded-l-3xl shadow-2xl flex flex-col">
              <div className="flex items-center justify-between p-5 border-b border-white/10">
                <div>
                  <p className="text-xs tracking-[0.3em] uppercase text-black">Filtre</p>
                  <p className="text-lg font-semibold text-white">Vyberte segment</p>
                </div>
                <Button variant="ghost" size="icon" onClick={onToggle} iconName="X" />
              </div>
              <div className="flex-1 overflow-y-auto p-5 space-y-6">
                <div>
                  <p className="text-[10px] uppercase tracking-[0.45em] text-black">Typ</p>
                  <div className="mt-3">{renderPropertyButtons("justify-start")}</div>
                </div>
                {renderPriceBlock()}
              </div>
              <div className="p-5 border-t border-white/10 flex gap-3">
                <Button variant="ghost" className="flex-1" onClick={onClearFilters}>
                  Resetovať
                </Button>
                <Button className="flex-1" onClick={onToggle}>
                  Hotovo
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default FilterControls;
