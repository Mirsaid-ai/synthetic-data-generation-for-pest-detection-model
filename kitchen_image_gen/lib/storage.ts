import fs from "fs";
import path from "path";

const STORAGE_DIR = path.join(process.cwd(), "public", "approved_images");

export interface ImageMetadata {
  id: string;
  prompt: string;
  filename: string;
  timestamp: string;
  width?: number;
  height?: number;
}

function ensureDir() {
  if (!fs.existsSync(STORAGE_DIR)) {
    fs.mkdirSync(STORAGE_DIR, { recursive: true });
  }
}

export function saveImage(
  id: string,
  imageBuffer: Buffer,
  prompt: string
): ImageMetadata {
  ensureDir();
  const filename = `kitchen_${id}.png`;
  const imagePath = path.join(STORAGE_DIR, filename);
  fs.writeFileSync(imagePath, imageBuffer);

  const metadata: ImageMetadata = {
    id,
    prompt,
    filename,
    timestamp: new Date().toISOString(),
  };
  const metaPath = path.join(STORAGE_DIR, `kitchen_${id}.json`);
  fs.writeFileSync(metaPath, JSON.stringify(metadata, null, 2));

  return metadata;
}

export function listImages(): ImageMetadata[] {
  ensureDir();
  const files = fs.readdirSync(STORAGE_DIR).filter((f) => f.endsWith(".json"));
  return files
    .map((f) => {
      try {
        const raw = fs.readFileSync(path.join(STORAGE_DIR, f), "utf-8");
        return JSON.parse(raw) as ImageMetadata;
      } catch {
        return null;
      }
    })
    .filter((m): m is ImageMetadata => m !== null)
    .sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
}

export function deleteImage(id: string): boolean {
  ensureDir();
  const imagePath = path.join(STORAGE_DIR, `kitchen_${id}.png`);
  const metaPath = path.join(STORAGE_DIR, `kitchen_${id}.json`);
  let deleted = false;
  if (fs.existsSync(imagePath)) {
    fs.unlinkSync(imagePath);
    deleted = true;
  }
  if (fs.existsSync(metaPath)) {
    fs.unlinkSync(metaPath);
    deleted = true;
  }
  return deleted;
}

export function getStorageDir(): string {
  ensureDir();
  return STORAGE_DIR;
}
