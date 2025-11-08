
import React from "react";

const ICON_PATH =
  "M12 2C6.486 2 2 6.486 2 12s4.486 10 10 10 10-4.486 10-10S17.514 2 12 2z";

function createIcon(displayName) {
  const Icon = React.forwardRef(({ size = 16, className = "", ...props }, ref) => (
    <svg
      ref={ref}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      <path d={ICON_PATH} />
      <text x="12" y="16" textAnchor="middle" fontSize="8">
        {displayName.slice(0, 1)}
      </text>
    </svg>
  ));
  Icon.displayName = displayName;
  return Icon;
}

export const MapPin = createIcon("MapPin");
export const Search = createIcon("Search");
export const Heart = createIcon("Heart");
export const Home = createIcon("Home");
export const Building = createIcon("Building");
export const Briefcase = createIcon("Briefcase");
export const Loader2 = createIcon("Loader2");
export const AlertCircle = createIcon("AlertCircle");
export const AlertTriangle = createIcon("AlertTriangle");
export const ChevronLeft = createIcon("ChevronLeft");
export const ChevronRight = createIcon("ChevronRight");
export const ChevronDown = createIcon("ChevronDown");
export const Check = createIcon("Check");
export const X = createIcon("X");
export const List = createIcon("List");
export const Map = createIcon("Map");
export const ZoomIn = createIcon("ZoomIn");
export const ZoomOut = createIcon("ZoomOut");
export const Star = createIcon("Star");
export const Users = createIcon("Users");
export const Bed = createIcon("Bed");
export const Bath = createIcon("Bath");
export const Share = createIcon("Share");
export const Filter = createIcon("Filter");
export const Eye = createIcon("Eye");
export const RotateCcw = createIcon("RotateCcw");
export const Calculator = createIcon("Calculator");
export const HelpCircle = createIcon("HelpCircle");
export const Maximize = createIcon("Maximize");
export const Minimize = createIcon("Minimize");
export const Plus = createIcon("Plus");
export const Minus = createIcon("Minus");
export const ExternalLink = createIcon("ExternalLink");

const exported = {
  MapPin,
  Search,
  Heart,
  Home,
  Building,
  Briefcase,
  Loader2,
  AlertCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Check,
  X,
  List,
  Map,
  ZoomIn,
  ZoomOut,
  Star,
  Users,
  Bed,
  Bath,
  Share,
  Filter,
  Eye,
  RotateCcw,
  Calculator,
  HelpCircle,
  Maximize,
  Minimize,
  Plus,
  Minus,
  ExternalLink,
};

export default exported;
