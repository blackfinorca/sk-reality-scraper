import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

export function getTaggerPlugin() {
  try {
    const taggerModule = require("@dhiwise/component-tagger");
    if (typeof taggerModule === "function") {
      return taggerModule();
    }
    if (taggerModule && typeof taggerModule.default === "function") {
      return taggerModule.default();
    }
    return taggerModule ?? { name: "component-tagger" };
  } catch (error) {
    console.warn(
      "[vite] @dhiwise/component-tagger not available; continuing with fallback plugin."
    );
    return {
      name: "component-tagger-fallback",
    };
  }
}
