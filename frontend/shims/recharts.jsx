
import React from "react";

function createContainer(name) {
  const Component = ({ children, ...props }) => (
    <div data-recharts={name} {...props}>
      {children}
    </div>
  );
  Component.displayName = name;
  return Component;
}

export const ResponsiveContainer = createContainer("ResponsiveContainer");
export const LineChart = createContainer("LineChart");
export const Line = createContainer("Line");
export const CartesianGrid = createContainer("CartesianGrid");
export const XAxis = createContainer("XAxis");
export const YAxis = createContainer("YAxis");
export const Tooltip = createContainer("Tooltip");
export const Legend = createContainer("Legend");
export const BarChart = createContainer("BarChart");
export const Bar = createContainer("Bar");
