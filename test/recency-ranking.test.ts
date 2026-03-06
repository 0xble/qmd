import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { applyRecencyBoost, computeRecencyFactor, createStore, hashContent, type Store } from "../src/store.js";
import type { CollectionConfig } from "../src/collections.js";

describe("recency ranking", () => {
  let testDir: string;
  let configDir: string;
  let dbPath: string;
  let store: Store | null = null;

  beforeEach(async () => {
    testDir = await mkdtemp(join(tmpdir(), "qmd-recency-"));
    configDir = join(testDir, "config");
    dbPath = join(testDir, "index.sqlite");
    process.env.QMD_CONFIG_DIR = configDir;
    await mkdir(configDir, { recursive: true });

    const config: CollectionConfig = {
      collections: {
        notes: {
          path: "/tmp/notes",
          pattern: "**/*.md",
        },
      },
    };
    await writeFile(join(configDir, "index.yml"), YAML.stringify(config), "utf-8");
  });

  afterEach(async () => {
    if (store) {
      store.close();
      store = null;
    }
    delete process.env.QMD_CONFIG_DIR;
    await rm(testDir, { recursive: true, force: true });
  });

  test("computeRecencyFactor decays by half-life", () => {
    const nowMs = Date.UTC(2026, 2, 6, 0, 0, 0); // 2026-03-06
    const recent = computeRecencyFactor("2026-03-06T00:00:00.000Z", { nowMs, halfLifeDays: 14 });
    const halfLifeAgo = computeRecencyFactor("2026-02-20T00:00:00.000Z", { nowMs, halfLifeDays: 14 });
    const old = computeRecencyFactor("2025-12-01T00:00:00.000Z", { nowMs, halfLifeDays: 14 });

    expect(recent).toBeCloseTo(1, 6);
    expect(halfLifeAgo).toBeCloseTo(0.5, 2);
    expect(old).toBeLessThan(halfLifeAgo);
  });

  test("applyRecencyBoost keeps base score when date is missing", () => {
    expect(applyRecencyBoost(0.42, null)).toBeCloseTo(0.42, 6);
    expect(applyRecencyBoost(0.42, undefined)).toBeCloseTo(0.42, 6);
  });

  test("searchFTS prefers newer docs when lexical relevance is equal", async () => {
    store = createStore(dbPath);
    const now = "2026-03-06T00:00:00.000Z";
    const sharedBody = "# Release Notes\n\nroadmap milestone planning";

    const oldHash = await hashContent(`${sharedBody}\nold`);
    store.insertContent(oldHash, `${sharedBody}\nold`, now);
    store.insertDocument(
      "notes",
      "old.md",
      "Old",
      oldHash,
      now,
      now,
      "2024-01-01T00:00:00.000Z"
    );

    const newHash = await hashContent(`${sharedBody}\nnew`);
    store.insertContent(newHash, `${sharedBody}\nnew`, now);
    store.insertDocument(
      "notes",
      "new.md",
      "New",
      newHash,
      now,
      now,
      "2026-03-01T00:00:00.000Z"
    );

    const results = store.searchFTS("roadmap milestone", 10, "notes");
    expect(results.length).toBeGreaterThanOrEqual(2);
    expect(results[0]?.filepath).toContain("/new.md");
    expect(results[1]?.filepath).toContain("/old.md");
    expect((results[0]?.score ?? 0)).toBeGreaterThan(results[1]?.score ?? 0);
  });
});
