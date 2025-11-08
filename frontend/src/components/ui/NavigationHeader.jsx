import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Icon from '../AppIcon';

const NavigationHeader = ({ className = '' }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navigationItems = [
    {
      label: 'Nástroje',
      path: '/toolbox-dashboard',
      icon: 'Calculator',
      tooltip: 'Investičné kalkulátory a analýzy'
    },
    {
      label: 'Mapa',
      path: '/map-view',
      icon: 'Map',
      tooltip: 'Interaktívna mapa nehnuteľností'
    }
  ];

  const handleNavigate = (path) => {
    navigate(path);
    setIsMobileMenuOpen(false);
  };

  const isActive = (path) => location?.pathname === path;

  return (
    <>
      <header
        className={`fixed top-0 left-0 right-0 z-50 bg-white/60 backdrop-blur-2xl border-b border-white/30 shadow-[0_12px_40px_rgba(15,23,42,0.08)] ${className}`}
      >
        <div className="flex items-center justify-between h-16 px-6 lg:px-10">
          <div />

          <nav className="hidden md:flex items-center space-x-2 bg-white/40 border border-white/50 rounded-full px-2 py-1 shadow-inner">
            {navigationItems?.map((item) => {
              const active = isActive(item?.path);
              return (
                <button
                  key={item?.path}
                  onClick={() => handleNavigate(item?.path)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 ${
                    active
                      ? "bg-white shadow-sm text-blue-700"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                  title={item?.tooltip}
                >
                  <Icon name={item?.icon} size={16} />
                  {item?.label}
                </button>
              );
            })}
          </nav>

          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded-full border border-slate-200 text-slate-600"
            aria-label="Toggle navigation menu"
          >
            <Icon name={isMobileMenuOpen ? "X" : "Menu"} size={22} />
          </button>
        </div>

        {isMobileMenuOpen && (
          <div className="md:hidden bg-white/95 backdrop-blur-xl border-t border-slate-200 shadow-lg">
            <nav className="px-6 py-4 space-y-2">
              {navigationItems?.map((item) => {
                const active = isActive(item?.path);
                return (
                  <button
                    key={item?.path}
                    onClick={() => handleNavigate(item?.path)}
                    className={`flex items-center w-full px-4 py-3 rounded-2xl text-sm font-semibold transition-all ${
                      active
                        ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg"
                        : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    <Icon name={item?.icon} size={18} className="mr-3" />
                    {item?.label}
                  </button>
                );
              })}
            </nav>
          </div>
        )}
      </header>

      <nav className="md:hidden fixed bottom-4 left-1/2 -translate-x-1/2 z-50 bg-white/90 backdrop-blur-xl border border-white/60 rounded-full shadow-[0_10px_35px_rgba(15,23,42,0.15)] px-3 py-2 flex space-x-4">
        {navigationItems?.map((item) => {
          const active = isActive(item?.path);
          return (
            <button
              key={item?.path}
              onClick={() => handleNavigate(item?.path)}
              className={`flex flex-col items-center text-xs font-semibold transition ${
                active ? "text-blue-600" : "text-slate-400"
              }`}
              title={item?.tooltip}
            >
              <span
                className={`p-2 rounded-full ${
                  active
                    ? "bg-gradient-to-br from-blue-600 to-blue-700 text-white"
                    : "bg-slate-100 text-slate-500"
                }`}
              >
                <Icon name={item?.icon} size={18} />
              </span>
              <span className="mt-1">{item?.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="h-20" />
    </>
  );
};

export default NavigationHeader;
