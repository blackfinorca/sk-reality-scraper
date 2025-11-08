import React from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const PortfolioSummary = ({ portfolioData, onViewAllProperties, className = '' }) => {
  return (
    <div className={`bg-card border border-border rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-foreground">Prehľad portfólia</h2>
        <Button 
          variant="outline" 
          onClick={onViewAllProperties}
          iconName="ExternalLink"
          iconPosition="right"
        >
          Zobraziť všetky
        </Button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {portfolioData?.map((item, index) => (
          <div key={index} className="text-center">
            <div className="flex items-center justify-center w-16 h-16 bg-primary/10 rounded-full mx-auto mb-3">
              <Icon name={item?.icon} size={24} className="text-primary" />
            </div>
            <div className="text-3xl font-bold text-foreground mb-1">{item?.value}</div>
            <div className="text-sm text-muted-foreground mb-2">{item?.label}</div>
            {item?.change && (
              <div className={`text-xs flex items-center justify-center space-x-1 ${
                item?.changeType === 'positive' ? 'text-success' : 'text-destructive'
              }`}>
                <Icon 
                  name={item?.changeType === 'positive' ? 'TrendingUp' : 'TrendingDown'} 
                  size={12} 
                />
                <span>{item?.change}</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default PortfolioSummary;