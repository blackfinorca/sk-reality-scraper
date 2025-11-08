function normalize(input) {
  if (!input) return [];
  if (typeof input === "string") return [input];
  if (Array.isArray(input)) {
    return input.flatMap(normalize);
  }
  if (typeof input === "object") {
    return Object.entries(input)
      .filter(([, value]) => Boolean(value))
      .map(([key]) => key);
  }
  return [String(input)];
}

export function cn(...inputs) {
  return normalize(inputs)
    .filter((value) => typeof value === "string" && value.trim().length > 0)
    .join(" ")
    .trim();
}
