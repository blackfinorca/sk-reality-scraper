import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import NavigationHeader from '../../components/ui/NavigationHeader';
import CalculatorCard from './components/CalculatorCard';
import MarketStatsCard from './components/MarketStatsCard';
import PortfolioSummary from './components/PortfolioSummary';
import MarketTrendsChart from './components/MarketTrendsChart';
import QuickActionsPanel from './components/QuickActionsPanel';
import RecentCalculationsPanel from './components/RecentCalculationsPanel';
import Button from '../../components/ui/Button';
import Icon from '../../components/AppIcon';

const ToolboxDashboard = () => {
  const navigate = useNavigate();
  const [currentLanguage, setCurrentLanguage] = useState('sk');

  useEffect(() => {
    const savedLanguage = localStorage.getItem('selectedLanguage') || 'sk';
    setCurrentLanguage(savedLanguage);
  }, []);

  // Mock data for calculator cards
  const calculatorCards = [
    {
      title: "Ohodnotenie nehnuteľnosti",
      description: "Analýza trhovej hodnoty",
      iconName: "Home",
      metrics: [
        { label: "Priemerná hodnota", value: "€185,000" },
        { label: "ROI potenciál", value: "7.2%" }
      ],
      lastUsed: "Dnes 14:30"
    },
    {
      title: "Buy & Hold analýza",
      description: "Dlhodobé prenájmy",
      iconName: "Building",
      metrics: [
        { label: "Mesačný príjem", value: "€850" },
        { label: "Cap Rate", value: "5.8%" }
      ],
      lastUsed: "Včera 16:45"
    },
    {
      title: "Flipping kalkulátor",
      description: "Rýchly predaj s renováciou",
      iconName: "RefreshCw",
      metrics: [
        { label: "Očakávaný zisk", value: "€25,000" },
        { label: "Doba realizácie", value: "6 mes." }
      ],
      lastUsed: "2 dni"
    },
    {
      title: "Cash Flow projektor",
      description: "Analýza peňažných tokov",
      iconName: "DollarSign",
      metrics: [
        { label: "Mesačný CF", value: "€320" },
        { label: "Ročný výnos", value: "€3,840" }
      ],
      lastUsed: "Minulý týždeň"
    }
  ];

  // Mock data for market statistics
  const marketStats = [
    {
      title: "Priemerná cena m²",
      value: "€2,450",
      change: "+3.2%",
      changeType: "positive",
      iconName: "TrendingUp",
      description: "Bratislava centrum"
    },
    {
      title: "Priemerný Cap Rate",
      value: "5.8%",
      change: "-0.3%",
      changeType: "negative",
      iconName: "Percent",
      description: "Slovenský trh"
    },
    {
      title: "Dostupné nehnuteľnosti",
      value: "1,247",
      change: "+12",
      changeType: "positive",
      iconName: "Building2",
      description: "Nové za týždeň"
    },
    {
      title: "Priemerná rentabilita",
      value: "6.4%",
      change: "+0.8%",
      changeType: "positive",
      iconName: "Target",
      description: "Investičné nehnuteľnosti"
    }
  ];

  // Mock data for portfolio summary
  const portfolioData = [
    {
      icon: "Bookmark",
      value: "23",
      label: "Uložené nehnuteľnosti",
      change: "+5",
      changeType: "positive"
    },
    {
      icon: "Eye",
      value: "8",
      label: "Sledované položky",
      change: "+2",
      changeType: "positive"
    },
    {
      icon: "Calculator",
      value: "47",
      label: "Dokončené analýzy",
      change: "+12",
      changeType: "positive"
    }
  ];

  // Mock data for market trends chart
  const priceChartData = [
    { month: 'Jan', price: 2200 },
    { month: 'Feb', price: 2250 },
    { month: 'Mar', price: 2300 },
    { month: 'Apr', price: 2280 },
    { month: 'May', price: 2350 },
    { month: 'Jun', price: 2400 },
    { month: 'Jul', price: 2420 },
    { month: 'Aug', price: 2450 }
  ];

  const capRateChartData = [
    { region: 'Bratislava', capRate: 5.2 },
    { region: 'Košice', capRate: 6.8 },
    { region: 'Prešov', capRate: 7.1 },
    { region: 'Žilina', capRate: 6.5 },
    { region: 'Banská Bystrica', capRate: 6.9 },
    { region: 'Nitra', capRate: 6.3 }
  ];

  // Mock data for recent calculations
  const recentCalculations = [
    {
      propertyName: "Byt 3+1, Bratislava - Ružinov",
      calculationType: "Buy & Hold analýza",
      type: "rental",
      date: "2025-11-07",
      result: "€420/mes",
      profitability: "positive",
      profitabilityText: "Pozitívny CF"
    },
    {
      propertyName: "Rodinný dom, Košice - Sever",
      calculationType: "Ohodnotenie",
      type: "valuation",
      date: "2025-11-06",
      result: "€165,000",
      profitability: "positive",
      profitabilityText: "Pod trhom"
    },
    {
      propertyName: "Byt 2+1, Žilina - Centrum",
      calculationType: "Flipping ROI",
      type: "flip",
      date: "2025-11-05",
      result: "€18,500",
      profitability: "positive",
      profitabilityText: "22% ROI"
    },
    {
      propertyName: "Komerčný priestor, Prešov",
      calculationType: "Cash Flow",
      type: "cashflow",
      date: "2025-11-04",
      result: "€680/mes",
      profitability: "positive",
      profitabilityText: "Vysoký výnos"
    }
  ];

  const handleCalculate = (calculatorType) => {
    console.log(`Opening ${calculatorType} calculator`);
    // In a real app, this would open a modal or navigate to calculator page
  };

  const handleViewDetails = (calculatorType) => {
    console.log(`Viewing details for ${calculatorType}`);
    // In a real app, this would show detailed analysis
  };

  const handleNewAnalysis = () => {
    console.log('Starting new analysis');
    // In a real app, this would open analysis wizard
  };

  const handleViewAllProperties = () => {
    navigate('/map-view');
  };

  const handleExportReport = () => {
    console.log('Exporting investment report');
    // In a real app, this would generate and download report
  };

  const handleViewAllCalculations = () => {
    console.log('Viewing all calculations');
    // In a real app, this would navigate to calculations history
  };

  return (
    <div className="min-h-screen bg-background">
      <NavigationHeader />
      <main className="container mx-auto px-6 py-8 max-w-7xl">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-foreground mb-2">
                Investičné nástroje
              </h1>
              <p className="text-muted-foreground">
                Komplexné analytické nástroje pre slovenský realitný trh
              </p>
            </div>
            <div className="flex items-center space-x-3 mt-4 md:mt-0">
              <Button 
                variant="outline"
                iconName="RefreshCw"
                iconPosition="left"
              >
                Aktualizovať údaje
              </Button>
              <Button 
                variant="default"
                onClick={handleNewAnalysis}
                iconName="Plus"
                iconPosition="left"
              >
                Nová analýza
              </Button>
            </div>
          </div>

          {/* Portfolio Summary */}
          <PortfolioSummary 
            portfolioData={portfolioData}
            onViewAllProperties={handleViewAllProperties}
            className="mb-8"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Calculator Cards */}
          <div className="lg:col-span-2 space-y-6">
            {/* Calculator Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {calculatorCards?.map((card, index) => (
                <CalculatorCard
                  key={index}
                  title={card?.title}
                  description={card?.description}
                  iconName={card?.iconName}
                  metrics={card?.metrics}
                  lastUsed={card?.lastUsed}
                  onCalculate={() => handleCalculate(card?.title)}
                  onViewDetails={() => handleViewDetails(card?.title)}
                />
              ))}
            </div>

            {/* Market Trends Charts */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <MarketTrendsChart
                chartData={priceChartData}
                chartType="line"
                title="Vývoj cien nehnuteľností"
              />
              <MarketTrendsChart
                chartData={capRateChartData}
                chartType="bar"
                title="Cap Rate podľa regiónov"
              />
            </div>
          </div>

          {/* Right Column - Side Panels */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <QuickActionsPanel
              onNewAnalysis={handleNewAnalysis}
              onViewProperties={handleViewAllProperties}
              onExportReport={handleExportReport}
            />

            {/* Market Statistics */}
            <div className="bg-card border border-border rounded-lg p-6">
              <div className="flex items-center space-x-2 mb-6">
                <Icon name="BarChart3" size={20} className="text-primary" />
                <h3 className="text-lg font-semibold text-foreground">Trhové štatistiky</h3>
              </div>
              <div className="space-y-4">
                {marketStats?.map((stat, index) => (
                  <MarketStatsCard
                    key={index}
                    title={stat?.title}
                    value={stat?.value}
                    change={stat?.change}
                    changeType={stat?.changeType}
                    iconName={stat?.iconName}
                    description={stat?.description}
                  />
                ))}
              </div>
            </div>

            {/* Recent Calculations */}
            <RecentCalculationsPanel
              calculations={recentCalculations}
              onViewAll={handleViewAllCalculations}
            />
          </div>
        </div>

        {/* Bottom Action Bar */}
        <div className="mt-12 bg-card border border-border rounded-lg p-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div className="mb-4 md:mb-0">
              <h3 className="text-lg font-semibold text-foreground mb-1">
                Pripravený na investovanie?
              </h3>
              <p className="text-sm text-muted-foreground">
                Preskúmajte dostupné nehnuteľnosti na interaktívnej mape
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <Button 
                variant="outline"
                iconName="BookOpen"
                iconPosition="left"
              >
                Sprievodca
              </Button>
              <Button 
                variant="default"
                onClick={handleViewAllProperties}
                iconName="Map"
                iconPosition="left"
              >
                Otvoriť mapu
              </Button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ToolboxDashboard;