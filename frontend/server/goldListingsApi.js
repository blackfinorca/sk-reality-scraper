import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = path.resolve(__dirname, "..");
const PROJECT_ROOT = path.resolve(FRONTEND_ROOT, "..");

const DEFAULT_PARQUET_CANDIDATES = [
  path.resolve(PROJECT_ROOT, "dedupe_runs/latest/gold_listings.parquet"),
  path.resolve(PROJECT_ROOT, "b2/realestate/gold/gold_listings_latest.parquet"),
  path.resolve(PROJECT_ROOT, "parquet_runs/latest/gold_listings.parquet"),
];

const FALLBACK_IMAGES = [
  "https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg",
  "https://images.pexels.com/photos/1643383/pexels-photo-1643383.jpeg",
  "https://images.pexels.com/photos/1571453/pexels-photo-1571453.jpeg",
  "https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg",
  "https://images.pexels.com/photos/1396122/pexels-photo-1396122.jpeg",
  "https://images.pexels.com/photos/2724749/pexels-photo-2724749.jpeg",
  "https://images.pexels.com/photos/2724748/pexels-photo-2724748.jpeg",
];

const cache = {
  normalized: null,
  filePath: null,
  mtimeMs: 0,
  meta: null,
};

function resolveParquetPath() {
  const envPath = process.env.GOLD_PARQUET_PATH;
  if (envPath) {
    const resolved = path.resolve(PROJECT_ROOT, envPath);
    if (fs.existsSync(resolved)) {
      return resolved;
    }
  }
  for (const candidate of DEFAULT_PARQUET_CANDIDATES) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(
    `Gold parquet file not found. Looked for ${[envPath, ...DEFAULT_PARQUET_CANDIDATES]
      .filter(Boolean)
      .join(", ")}`
  );
}

async function readParquetRows(parquetPath) {
  const scriptPath = path.resolve(PROJECT_ROOT, "scripts", "export_listings_snapshot.py");
  const limit = process.env.GOLD_LISTINGS_LIMIT || "500";
  const { stdout } = await execFileAsync(
    "python3",
    [scriptPath, "--parquet-path", parquetPath, "--limit", String(limit)],
    {
      cwd: PROJECT_ROOT,
      maxBuffer: 10 * 1024 * 1024,
    }
  );
  const payload = JSON.parse(stdout);
  const rows = Array.isArray(payload?.data) ? payload.data : payload;
  return { rows, meta: payload?.meta || {} };
}

function toNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function slugifyRegion(value) {
  if (!value) return "unknown";
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function deriveInvestmentScore(grade, capRate) {
  if (grade === "A") return 8.5;
  if (grade === "B") return 6.5;
  if (grade === "C") return 4.5;
  if (capRate >= 6) return 8;
  if (capRate >= 4) return 6;
  return 4;
}

function pickImages(id) {
  if (!FALLBACK_IMAGES.length) return [];
  const hash = Math.abs(
    Array.from(String(id || "seed")).reduce((acc, char) => acc + char.charCodeAt(0), 0)
  );
  const images = [];
  for (let i = 0; i < 3; i += 1) {
    const index = (hash + i) % FALLBACK_IMAGES.length;
    images.push(FALLBACK_IMAGES[index]);
  }
  return images;
}

function normalizeRecord(record) {
  if (!record) return null;

  const listingId = record.listing_id || record.listing_uid || record.id;
  if (!listingId) return null;

  const price = toNumber(record.price);
  const size = toNumber(record.size_m2);
  const pricePerSqm = toNumber(record.price_psm) || (price && size ? Math.round(price / size) : null);
  const monthlyRent = price ? Math.round(price * 0.004) : null;
  const monthlyExpenses = price ? Math.round(price * 0.0009) : null;
  const cashFlow =
    monthlyRent !== null && monthlyExpenses !== null ? monthlyRent - monthlyExpenses : null;
  const capRate =
    price && monthlyRent
      ? Number((((monthlyRent * 12) / price) * 100).toFixed(1))
      : null;
  const roi = capRate !== null ? Number((capRate * 1.6).toFixed(1)) : null;

  const grade =
    record.grade ||
    (capRate !== null
      ? capRate >= 6
        ? "A"
        : capRate >= 4
          ? "B"
          : "C"
      : "B");
  const investmentScore = deriveInvestmentScore(grade, capRate ?? 0);

  const lat = toNumber(record.lat);
  const lng = toNumber(record.lng);
  const coordinates = lat !== null && lng !== null ? { lat, lng } : null;

  const rawTitle = record.title || record.project_name_norm;
  const title =
    rawTitle ||
    `Ponuka v ${record.address_town || record.address_ascii || record.address_norm || "Slovensko"}`;
  const city = record.address_town || record.address_ascii || "Neznáme mesto";
  const regionSlug = slugifyRegion(record.region || record.address_town || "");
  const type =
    record.property_type ||
    (record.rooms && record.rooms >= 4 ? "house" : "apartment");

  const summary = record.summary_short_sk || null;

  const amenities = [
    record.status ? `Status: ${record.status}` : null,
    record.energ_cert ? `Energetická trieda ${record.energ_cert}` : null,
    record.primary_portal ? `Portal: ${record.primary_portal}` : null,
  ].filter(Boolean);

  return {
    id: listingId,
    listingId,
    title,
    location: `${city}${record.address_norm ? ` • ${record.address_norm}` : ""}`,
    address: record.address_norm || record.address_ascii || city,
    city,
    region: regionSlug || "unknown",
    price,
    pricePerSqm,
    monthlyRent,
    monthlyExpenses,
    cashFlow,
    capRate,
    roi,
    type,
    area: size,
    bedrooms: toNumber(record.rooms),
    bathrooms: toNumber(record.bathrooms) || 1,
    yearBuilt: toNumber(record.year_of_construction),
    condition: record.status || "unknown",
    grade,
    investmentScore,
    coordinates,
    lat,
    lng,
    url: record.primary_source_url || record.url,
    portal: record.primary_portal,
    summary,
    description: summary,
    amenities,
    images: record.images && record.images.length ? record.images : pickImages(listingId),
    imageAlts: [],
    sourceUpdatedAt: record.updated_at,
  };
}

async function ensureCache() {
  const parquetPath = resolveParquetPath();
  const stats = await fs.promises.stat(parquetPath);
  if (cache.normalized && cache.mtimeMs === stats.mtimeMs && cache.filePath === parquetPath) {
    return cache;
  }
  const { rows, meta } = await readParquetRows(parquetPath);
  cache.normalized = rows.map(normalizeRecord).filter(Boolean);
  cache.filePath = parquetPath;
  cache.mtimeMs = stats.mtimeMs;
  cache.meta = { ...meta, parquet_path: parquetPath };
  return cache;
}

function filterByQuery(records, params) {
  const city = params.get("city");
  const region = params.get("region");
  const search = params.get("q");

  return records.filter((record) => {
    if (city && record.city?.toLowerCase() !== city.toLowerCase()) {
      return false;
    }
    if (region && record.region !== region) {
      return false;
    }
    if (search) {
      const needle = search.toLowerCase();
      const haystack = `${record.title} ${record.location} ${record.summary || ""}`.toLowerCase();
      if (!haystack.includes(needle)) {
        return false;
      }
    }
    return true;
  });
}

export function goldListingsApiPlugin() {
  return {
    name: "gold-listings-api",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        if (!req.url.startsWith("/api/listings")) {
          return next();
        }
        if (req.method !== "GET") {
          res.statusCode = 405;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "Method not allowed" }));
          return;
        }
        try {
          const { normalized, meta } = await ensureCache();
          const url = new URL(req.url, "http://localhost");
          const limit = Number(url.searchParams.get("limit") || "0");
          const filtered = filterByQuery(normalized, url.searchParams);
          const sliced = limit > 0 ? filtered.slice(0, limit) : filtered;
          res.setHeader("Content-Type", "application/json");
          res.end(
            JSON.stringify({
              data: sliced,
              meta: {
                ...meta,
                total: filtered.length,
                limit: limit > 0 ? limit : filtered.length,
              },
            })
          );
        } catch (error) {
          console.error("[gold-listings-api] failed", error);
          res.statusCode = 500;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: error.message }));
        }
      });
    },
  };
}
