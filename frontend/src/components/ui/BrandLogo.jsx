import React from 'react';

const BrandLogo = ({ onClick, className = '' }) => {
  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-3 hover:opacity-80 transition-opacity duration-200 ease-out ${className}`}
      aria-label="SlovakInvest Pro - Domov"
    >
      {/* Logo Icon */}
      <div className="flex items-center justify-center w-10 h-10 bg-primary rounded-lg shadow-elevation-1">
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="text-primary-foreground"
        >
          <path
            d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path
            d="M9 22V12H15V22"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle
            cx="12"
            cy="8"
            r="2"
            fill="var(--color-accent)"
            stroke="none"
          />
        </svg>
      </div>

      {/* Brand Text */}
      <div className="flex flex-col items-start">
        <div className="text-lg font-semibold text-primary leading-tight">
          SlovakInvest
        </div>
        <div className="text-xs font-medium text-accent leading-tight">
          Pro
        </div>
      </div>
    </button>
  );
};

export default BrandLogo;