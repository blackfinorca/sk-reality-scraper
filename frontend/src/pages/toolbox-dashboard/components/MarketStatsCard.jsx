import React from 'react';
import Icon from '../../../components/AppIcon';

const MarketStatsCard = ({ 
  title, 
  value, 
  change, 
  changeType, 
  iconName, 
  description,
  className = '' 
}) => {
  const getChangeColor = () => {
    if (changeType === 'positive') return 'text-success';
    if (changeType === 'negative') return 'text-destructive';
    return 'text-muted-foreground';
  };

  const getChangeIcon = () => {
    if (changeType === 'positive') return 'TrendingUp';
    if (changeType === 'negative') return 'TrendingDown';
    return 'Minus';
  };

  return (
    <div className={`bg-card border border-border rounded-lg p-6 ${className}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <Icon name={iconName} size={20} className="text-primary" />
            <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          </div>
          
          <div className="text-2xl font-bold text-foreground mb-1">{value}</div>
          
          {change && (
            <div className={`flex items-center space-x-1 text-sm ${getChangeColor()}`}>
              <Icon name={getChangeIcon()} size={14} />
              <span>{change}</span>
            </div>
          )}
          
          {description && (
            <p className="text-xs text-muted-foreground mt-2">{description}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default MarketStatsCard;