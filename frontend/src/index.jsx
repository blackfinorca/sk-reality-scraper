import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles/tailwind.css";
import "./styles/index.css";

const container = document.getElementById("root");
const root = createRoot(container);

if (typeof window !== "undefined") {
  window.__COMPONENT_ERROR__ = (error, errorInfo) => {
    // eslint-disable-next-line no-console
    console.error("Component error:", error, errorInfo);
  };
}

root.render(<App />);
