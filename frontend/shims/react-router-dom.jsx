import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

const RouterContext = createContext({
  path: "/",
  navigate: () => {},
  location: { pathname: "/" },
});

export function BrowserRouter({ children }) {
  const [path, setPath] = useState(() => (typeof window !== "undefined" ? window.location.pathname : "/"));

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handler = () => setPath(window.location.pathname);
    window.addEventListener("popstate", handler);
    return () => window.removeEventListener("popstate", handler);
  }, []);

  const navigate = (to) => {
    if (typeof window === "undefined" || !to) return;
    if (to === window.location.pathname) return;
    window.history.pushState({}, "", to);
    setPath(window.location.pathname);
  };

  const value = useMemo(() => ({ path, navigate, location: { pathname: path } }), [path]);
  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

export function Routes({ children }) {
  const { path } = useContext(RouterContext);
  let element = null;
  React.Children.forEach(children, (child) => {
    if (element || !React.isValidElement(child)) return;
    const childPath = child.props.path || "/";
    if (matchPath(childPath, path)) {
      element = child.props.element || child.props.children || null;
    }
  });
  return element || null;
}

export function Route() {
  return null;
}

export function useNavigate() {
  const ctx = useContext(RouterContext);
  return ctx.navigate || (() => {});
}

export function useLocation() {
  const ctx = useContext(RouterContext);
  return ctx.location || { pathname: "/" };
}

export function Link({ to, children, ...props }) {
  const navigate = useNavigate();
  const handleClick = (event) => {
    event.preventDefault();
    navigate(to);
  };
  return (
    <a href={to} onClick={handleClick} {...props}>
      {children}
    </a>
  );
}

function matchPath(pattern, currentPath) {
  if (pattern === currentPath) return true;
  if (pattern === "*") return true;
  if (!pattern || pattern === "/") return currentPath === "/";
  return currentPath.startsWith(pattern);
}
