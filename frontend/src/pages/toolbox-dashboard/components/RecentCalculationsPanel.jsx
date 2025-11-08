import React from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const RecentCalculationsPanel = ({ calculations, onViewAll, className = '' }) => {
  const getCalculationIcon = (type) => {
    switch (type) {
      case 'valuation': return 'Home';
      case 'rental': return 'Building';
      case 'flip': return 'RefreshCw';
      case 'cashflow': return 'DollarSign';
      default: return 'Calculator';
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date?.toLocaleDateString('sk-SK', { 
      day: '2-digit', 
      month: '2-digit', 
      year: 'numeric' 
    });
  };

  return (
    <div className={`bg-card border border-border rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Icon name="History" size={20} className="text-primary" />
          <h3 className="text-lg font-semibold text-foreground">Nedávne výpočty</h3>
        </div>
        <Button 
          variant="ghost" 
          size="sm"
          onClick={onViewAll}
          iconName="ArrowRight"
          iconPosition="right"
        >
          Zobraziť všetky
        </Button>
      </div>
      <div className="space-y-3">
        {calculations?.length === 0 ? (
          <div className="text-center py-8">
            <Icon name="Calculator" size={48} className="text-muted-foreground mx-auto mb-3" />
            <p className="text-sm text-muted-foreground">Zatiaľ žiadne výpočty</p>
            <p className="text-xs text-muted-foreground">Začnite svoju prvú analýzu</p>
          </div>
        ) : (
          calculations?.map((calc, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors cursor-pointer">
              <div className="flex items-center space-x-3">
                <div className="flex items-center justify-center w-8 h-8 bg-primary/10 rounded-md">
                  <Icon name={getCalculationIcon(calc?.type)} size={16} className="text-primary" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-foreground">{calc?.propertyName}</h4>
                  <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                    <span>{calc?.calculationType}</span>
                    <span>•</span>
                    <span>{formatDate(calc?.date)}</span>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm font-semibold text-foreground">{calc?.result}</div>
                <div className={`text-xs ${calc?.profitability === 'positive' ? 'text-success' : calc?.profitability === 'negative' ? 'text-destructive' : 'text-muted-foreground'}`}>
                  {calc?.profitabilityText}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default RecentCalculationsPanel;