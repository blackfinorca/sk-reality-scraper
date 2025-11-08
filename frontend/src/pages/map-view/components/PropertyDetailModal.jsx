import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Image from '../../../components/AppImage';
import Button from '../../../components/ui/Button';

const PropertyDetailModal = ({ property, isOpen, onClose }) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [activeTab, setActiveTab] = useState('overview');

  if (!isOpen || !property) return null;

  const nextImage = () => {
    setCurrentImageIndex((prev) => 
      prev === property?.images?.length - 1 ? 0 : prev + 1
    );
  };

  const prevImage = () => {
    setCurrentImageIndex((prev) => 
      prev === 0 ? property?.images?.length - 1 : prev - 1
    );
  };

  const getPropertyTypeLabel = (type) => {
    const types = {
      apartment: 'Byt',
      house: 'Dom',
      commercial: 'Komerčné',
      land: 'Pozemok'
    };
    return types?.[type] || type;
  };

  const getInvestmentScoreColor = (score) => {
    if (score >= 8) return 'text-emerald-600';
    if (score >= 6) return 'text-amber-600';
    return 'text-rose-600';
  };

  const getInvestmentScoreBg = (score) => {
    if (score >= 8) return 'bg-emerald-50 border-emerald-200';
    if (score >= 6) return 'bg-amber-50 border-amber-200';
    return 'bg-rose-50 border-rose-200';
  };

  const tabs = [
    { id: 'overview', label: 'Prehľad', icon: 'Home' },
    { id: 'financial', label: 'Finančné', icon: 'Calculator' },
    { id: 'location', label: 'Lokalita', icon: 'MapPin' }
  ];

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm">
      <div className="fixed inset-4 bg-white rounded-2xl shadow-2xl overflow-hidden max-w-7xl mx-auto">
        {/* Modern Header with Gradient */}
        <div className="relative bg-gradient-to-r from-slate-50 to-gray-100 p-6 border-b border-gray-200">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-gray-900 mb-2 leading-tight">
                {property?.title}
              </h2>
              <div className="flex items-center text-gray-600 mb-3">
                <Icon name="MapPin" size={18} className="mr-2 text-gray-500" />
                <span className="text-lg">{property?.location}</span>
              </div>
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <Icon name="Star" size={16} className="text-amber-400 fill-current" />
                  <span className="font-medium text-gray-700">Nové</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Icon name="Users" size={16} className="text-gray-500" />
                  <span className="text-gray-600">{property?.bedrooms} hostia</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Icon name="Bed" size={16} className="text-gray-500" />
                  <span className="text-gray-600">{property?.bedrooms} spálne</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Icon name="Bath" size={16} className="text-gray-500" />
                  <span className="text-gray-600">{property?.bathrooms} kúpeľne</span>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`px-4 py-2 rounded-full border-2 ${getInvestmentScoreBg(property?.investmentScore)}`}>
                <span className={`text-sm font-bold ${getInvestmentScoreColor(property?.investmentScore)}`}>
                  Skóre {property?.investmentScore}/10
                </span>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-10 w-10 rounded-full hover:bg-gray-200 transition-colors"
                iconName="X"
              />
            </div>
          </div>
        </div>

        <div className="flex h-[calc(100vh-200px)]">
          {/* Enhanced Image Gallery - Airbnb Style */}
          <div className="w-3/5 relative bg-black">
            <div className="relative h-full">
              <Image
                src={property?.images?.[currentImageIndex]}
                alt={property?.imageAlts?.[currentImageIndex]}
                className="w-full h-full object-cover"
              />
              
              {/* Enhanced Navigation */}
              {property?.images?.length > 1 && (
                <>
                  <button
                    onClick={prevImage}
                    className="absolute left-6 top-1/2 transform -translate-y-1/2 p-3 bg-white/90 backdrop-blur-sm rounded-full hover:bg-white hover:scale-105 transition-all shadow-lg"
                  >
                    <Icon name="ChevronLeft" size={24} className="text-gray-800" />
                  </button>
                  <button
                    onClick={nextImage}
                    className="absolute right-6 top-1/2 transform -translate-y-1/2 p-3 bg-white/90 backdrop-blur-sm rounded-full hover:bg-white hover:scale-105 transition-all shadow-lg"
                  >
                    <Icon name="ChevronRight" size={24} className="text-gray-800" />
                  </button>
                </>
              )}

              {/* Modern Image Counter */}
              <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 px-4 py-2 bg-black/60 backdrop-blur-sm rounded-full text-white font-medium">
                {currentImageIndex + 1} / {property?.images?.length}
              </div>

              {/* Action Buttons Overlay */}
              <div className="absolute top-6 right-6 flex space-x-3">
                <button className="p-3 bg-white/90 backdrop-blur-sm rounded-full hover:bg-white transition-all shadow-lg">
                  <Icon name="Share" size={20} className="text-gray-800" />
                </button>
                <button className="p-3 bg-white/90 backdrop-blur-sm rounded-full hover:bg-white transition-all shadow-lg">
                  <Icon name="Heart" size={20} className="text-gray-800" />
                </button>
              </div>
            </div>

            {/* Enhanced Thumbnail Grid */}
            {property?.images?.length > 1 && (
              <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/80 to-transparent">
                <div className="flex space-x-3 overflow-x-auto pb-2">
                  {property?.images?.map((image, index) => (
                    <button
                      key={index}
                      onClick={() => setCurrentImageIndex(index)}
                      className={`flex-shrink-0 w-20 h-16 rounded-lg overflow-hidden border-3 transition-all ${
                        index === currentImageIndex 
                          ? 'border-white shadow-lg transform scale-105' 
                          : 'border-white/50 hover:border-white/80'
                      }`}
                    >
                      <Image
                        src={image}
                        alt={property?.imageAlts?.[index]}
                        className="w-full h-full object-cover"
                      />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Enhanced Content Panel */}
          <div className="w-2/5 flex flex-col bg-white">
            {/* Airbnb-style Tabs */}
            <div className="flex border-b border-gray-200 bg-gray-50">
              {tabs?.map((tab) => (
                <button
                  key={tab?.id}
                  onClick={() => setActiveTab(tab?.id)}
                  className={`flex items-center px-6 py-4 text-sm font-medium transition-all ${
                    activeTab === tab?.id
                      ? 'text-gray-900 border-b-3 border-gray-900 bg-white' :'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Icon name={tab?.icon} size={18} className="mr-3" />
                  {tab?.label}
                </button>
              ))}
            </div>

            {/* Enhanced Tab Content */}
            <div className="flex-1 overflow-y-auto p-8">
              {activeTab === 'overview' && (
                <div className="space-y-8">
                  {/* Modern Price Display */}
                  <div className="grid grid-cols-2 gap-6">
                    <div className="bg-gray-50 p-6 rounded-2xl">
                      <div className="text-3xl font-bold text-gray-900 mb-1">
                        €{property?.price?.toLocaleString('sk-SK')}
                      </div>
                      <div className="text-gray-600 text-lg">
                        €{property?.pricePerSqm}/m² • Celková cena
                      </div>
                    </div>
                    <div className="bg-emerald-50 p-6 rounded-2xl border border-emerald-200">
                      <div className="text-2xl font-bold text-emerald-700 mb-1">
                        €{property?.monthlyRent}
                      </div>
                      <div className="text-emerald-600 text-lg">
                        mesačný nájom
                      </div>
                    </div>
                  </div>

                  {/* Enhanced Property Details Grid */}
                  <div className="bg-gray-50 p-6 rounded-2xl">
                    <h4 className="font-bold text-gray-900 mb-4 text-lg">Detaily nehnuteľnosti</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="flex justify-between py-3 border-b border-gray-200">
                        <span className="text-gray-600 font-medium">Typ:</span>
                        <span className="font-semibold text-gray-900">{getPropertyTypeLabel(property?.type)}</span>
                      </div>
                      <div className="flex justify-between py-3 border-b border-gray-200">
                        <span className="text-gray-600 font-medium">Plocha:</span>
                        <span className="font-semibold text-gray-900">{property?.area} m²</span>
                      </div>
                      <div className="flex justify-between py-3 border-b border-gray-200">
                        <span className="text-gray-600 font-medium">Spálne:</span>
                        <span className="font-semibold text-gray-900">{property?.bedrooms}</span>
                      </div>
                      <div className="flex justify-between py-3 border-b border-gray-200">
                        <span className="text-gray-600 font-medium">Kúpeľne:</span>
                        <span className="font-semibold text-gray-900">{property?.bathrooms}</span>
                      </div>
                      <div className="flex justify-between py-3 border-b border-gray-200">
                        <span className="text-gray-600 font-medium">Rok výstavby:</span>
                        <span className="font-semibold text-gray-900">{property?.yearBuilt}</span>
                      </div>
                      <div className="flex justify-between py-3 border-b border-gray-200">
                        <span className="text-gray-600 font-medium">Stav:</span>
                        <span className="font-semibold text-gray-900">{property?.condition}</span>
                      </div>
                    </div>
                  </div>

                  {/* Enhanced Description */}
                  <div>
                    <h4 className="font-bold text-gray-900 mb-4 text-lg">O tejto nehnuteľnosti</h4>
                    <p className="text-gray-700 leading-relaxed text-base whitespace-pre-line">
                      {property?.description}
                    </p>
                  </div>
                </div>
              )}

              {activeTab === 'financial' && (
                <div className="space-y-8">
                  {/* Enhanced Metrics Cards */}
                  <div className="grid grid-cols-1 gap-4">
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-100 p-6 rounded-2xl border border-blue-200">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-3xl font-bold text-blue-700">{property?.capRate}%</div>
                          <div className="text-blue-600 font-medium">Cap Rate</div>
                        </div>
                        <Icon name="TrendingUp" size={32} className="text-blue-500" />
                      </div>
                    </div>
                    <div className="bg-gradient-to-br from-emerald-50 to-green-100 p-6 rounded-2xl border border-emerald-200">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-3xl font-bold text-emerald-700">{property?.roi}%</div>
                          <div className="text-emerald-600 font-medium">ROI</div>
                        </div>
                        <Icon name="DollarSign" size={32} className="text-emerald-500" />
                      </div>
                    </div>
                    <div className={`p-6 rounded-2xl border-2 ${property?.cashFlow > 0 ? 'bg-gradient-to-br from-emerald-50 to-green-100 border-emerald-200' : 'bg-gradient-to-br from-rose-50 to-red-100 border-rose-200'}`}>
                      <div className="flex items-center justify-between">
                        <div>
                          <div className={`text-3xl font-bold ${property?.cashFlow > 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                            {property?.cashFlow > 0 ? '+' : ''}€{property?.cashFlow}
                          </div>
                          <div className={`font-medium ${property?.cashFlow > 0 ? 'text-emerald-600' : 'text-rose-600'}`}>Cash Flow</div>
                        </div>
                        <Icon name="ArrowUpRight" size={32} className={property?.cashFlow > 0 ? 'text-emerald-500' : 'text-rose-500'} />
                      </div>
                    </div>
                  </div>

                  {/* Enhanced Financial Breakdown */}
                  <div className="bg-gray-50 p-6 rounded-2xl">
                    <h4 className="font-bold text-gray-900 mb-6 text-lg">Finančná analýza</h4>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center py-4 border-b border-gray-200">
                        <span className="text-gray-600 font-medium text-base">Mesačný nájom:</span>
                        <span className="font-bold text-gray-900 text-lg">€{property?.monthlyRent}</span>
                      </div>
                      <div className="flex justify-between items-center py-4 border-b border-gray-200">
                        <span className="text-gray-600 font-medium text-base">Ročný nájom:</span>
                        <span className="font-bold text-gray-900 text-lg">€{(property?.monthlyRent * 12)?.toLocaleString('sk-SK')}</span>
                      </div>
                      <div className="flex justify-between items-center py-4 border-b border-gray-200">
                        <span className="text-gray-600 font-medium text-base">Mesačné náklady:</span>
                        <span className="font-bold text-gray-900 text-lg">€{property?.monthlyExpenses}</span>
                      </div>
                      <div className="flex justify-between items-center py-4">
                        <span className="text-gray-600 font-medium text-base">Čistý cash flow:</span>
                        <span className={`font-bold text-lg ${property?.cashFlow > 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                          {property?.cashFlow > 0 ? '+' : ''}€{property?.cashFlow}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'location' && (
                <div className="space-y-8">
                  {/* Enhanced Map */}
                  <div className="h-80 bg-gray-200 rounded-2xl overflow-hidden shadow-inner">
                    <iframe
                      width="100%"
                      height="100%"
                      loading="lazy"
                      title={`Mapa lokality ${property?.location}`}
                      referrerPolicy="no-referrer-when-downgrade"
                      src={`https://www.google.com/maps?q=${property?.coordinates?.lat},${property?.coordinates?.lng}&z=15&output=embed`}
                      className="rounded-2xl"
                    />
                  </div>

                  {/* Enhanced Location Info */}
                  <div className="bg-gray-50 p-6 rounded-2xl">
                    <h4 className="font-bold text-gray-900 mb-4 text-lg">Informácie o lokalite</h4>
                    <div className="space-y-4">
                      <div className="flex justify-between py-3 border-b border-gray-200">
                        <span className="text-gray-600 font-medium">Adresa:</span>
                        <span className="font-semibold text-gray-900 text-right">{property?.address}</span>
                      </div>
                      <div className="flex justify-between py-3 border-b border-gray-200">
                        <span className="text-gray-600 font-medium">Región:</span>
                        <span className="font-semibold text-gray-900 capitalize">{property?.region}</span>
                      </div>
                      <div className="flex justify-between py-3">
                        <span className="text-gray-600 font-medium">Vzdialenosť do centra:</span>
                        <span className="font-semibold text-gray-900">{property?.distanceToCenter} km</span>
                      </div>
                    </div>
                  </div>

                  {/* Enhanced Amenities */}
                  <div>
                    <h4 className="font-bold text-gray-900 mb-4 text-lg">Čo ponúka okolie</h4>
                    <div className="grid grid-cols-1 gap-3">
                      {property?.amenities?.map((amenity, index) => (
                        <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-xl">
                          <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center">
                            <Icon name="Check" size={16} className="text-emerald-600" />
                          </div>
                          <span className="text-gray-800 font-medium">{amenity}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Enhanced Action Buttons - Airbnb Style */}
            <div className="p-8 border-t border-gray-200 bg-white">
              <div className="flex space-x-4">
                <Button
                  variant="outline"
                  iconName="Heart"
                  iconPosition="left"
                  className="flex-1 py-4 text-base font-semibold border-2 hover:bg-gray-50 rounded-xl"
                >
                  Uložiť
                </Button>
                <Button
                  variant="outline"
                  iconName="Calculator"
                  iconPosition="left"
                  className="flex-1 py-4 text-base font-semibold border-2 hover:bg-gray-50 rounded-xl"
                >
                  Analýza
                </Button>
                <Button
                  iconName="Phone"
                  iconPosition="left"
                  className="flex-1 py-4 text-base font-bold bg-gradient-to-r from-rose-500 to-pink-600 hover:from-rose-600 hover:to-pink-700 rounded-xl shadow-lg hover:shadow-xl transition-all"
                >
                  Kontaktovať
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PropertyDetailModal;