import React from 'react';
import Button from '../../../components/ui/Button';
import Icon from '../../../components/AppIcon';

const QuickActionsPanel = ({ onNewAnalysis, onViewProperties, onExportReport, className = '' }) => {
  const quickActions = [
    {
      title: "Nová analýza",
      description: "Spustiť novú investičnú analýzu",
      iconName: "Plus",
      variant: "default",
      onClick: onNewAnalysis
    },
    {
      title: "Mapa nehnuteľností",
      description: "Prehliadať dostupné nehnuteľnosti",
      iconName: "Map",
      variant: "outline",
      onClick: onViewProperties
    },
    {
      title: "Export reportu",
      description: "Stiahnuť investičný report",
      iconName: "Download",
      variant: "secondary",
      onClick: onExportReport
    }
  ];

  return (
    <div className={`bg-card border border-border rounded-lg p-6 ${className}`}>
      <div className="flex items-center space-x-2 mb-6">
        <Icon name="Zap" size={20} className="text-primary" />
        <h3 className="text-lg font-semibold text-foreground">Rýchle akcie</h3>
      </div>
      <div className="space-y-4">
        {quickActions?.map((action, index) => (
          <div key={index} className="flex items-center justify-between p-4 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-10 h-10 bg-primary/10 rounded-lg">
                <Icon name={action?.iconName} size={18} className="text-primary" />
              </div>
              <div>
                <h4 className="text-sm font-medium text-foreground">{action?.title}</h4>
                <p className="text-xs text-muted-foreground">{action?.description}</p>
              </div>
            </div>
            <Button 
              variant={action?.variant}
              size="sm"
              onClick={action?.onClick}
              iconName="ArrowRight"
              iconPosition="right"
            >
              Spustiť
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default QuickActionsPanel;