import React from 'react';
import Button from '../../../components/ui/Button';
import Icon from '../../../components/AppIcon';

const CalculatorCard = ({ 
  title, 
  description, 
  iconName, 
  metrics, 
  lastUsed, 
  onCalculate, 
  onViewDetails,
  className = '' 
}) => {
  return (
    <div className={`bg-card border border-border rounded-lg p-6 hover:shadow-lg transition-all duration-200 ${className}`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-12 h-12 bg-primary/10 rounded-lg">
            <Icon name={iconName} size={24} className="text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">{title}</h3>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        </div>
      </div>
      {/* Metrics */}
      {metrics && metrics?.length > 0 && (
        <div className="grid grid-cols-2 gap-4 mb-4">
          {metrics?.map((metric, index) => (
            <div key={index} className="text-center p-3 bg-muted/50 rounded-md">
              <div className="text-lg font-semibold text-foreground">{metric?.value}</div>
              <div className="text-xs text-muted-foreground">{metric?.label}</div>
            </div>
          ))}
        </div>
      )}
      {/* Last Used */}
      {lastUsed && (
        <div className="flex items-center text-xs text-muted-foreground mb-4">
          <Icon name="Clock" size={14} className="mr-1" />
          Naposledy použité: {lastUsed}
        </div>
      )}
      {/* Actions */}
      <div className="flex space-x-2">
        <Button 
          variant="default" 
          onClick={onCalculate}
          iconName="Calculator"
          iconPosition="left"
          className="flex-1"
        >
          Vypočítať
        </Button>
        <Button 
          variant="outline" 
          onClick={onViewDetails}
          iconName="Eye"
          size="default"
        >
        </Button>
      </div>
    </div>
  );
};

export default CalculatorCard;