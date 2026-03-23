"use client";

import { useEffect, useState } from "react";

interface ImageMeta {
  id: string;
  prompt: string;
  filename: string;
  timestamp: string;
}

export default function ImageGallery() {
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchImages = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/images");
      const data = await res.json();
      setImages(data.images || []);
    } catch {
      console.error("Failed to fetch images");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchImages();
  }, []);

  const deleteImage = async (id: string) => {
    if (!confirm("Delete this kitchen image?")) return;
    await fetch(`/api/images?id=${id}`, { method: "DELETE" });
    fetchImages();
  };

  if (loading) {
    return <p className="text-zinc-500">Loading gallery...</p>;
  }

  if (images.length === 0) {
    return (
      <div className="text-center py-16">
        <p className="text-zinc-500 text-lg">No approved images yet.</p>
        <p className="text-zinc-600 text-sm mt-2">
          Go to Generate to create kitchen images.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <p className="text-sm text-zinc-400">
          {images.length} approved image{images.length !== 1 ? "s" : ""}
        </p>
        <button
          onClick={fetchImages}
          className="text-sm text-zinc-500 hover:text-white transition-colors"
        >
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {images.map((img) => (
          <div
            key={img.id}
            className="group border border-zinc-800 rounded-lg overflow-hidden bg-zinc-900 hover:border-zinc-600 transition-colors"
          >
            <div className="aspect-video relative">
              <img
                src={`/approved_images/${img.filename}`}
                alt={img.prompt}
                className="w-full h-full object-cover"
              />
              <button
                onClick={() => deleteImage(img.id)}
                className="absolute top-2 right-2 p-1.5 bg-red-600/80 hover:bg-red-500 rounded-md opacity-0 group-hover:opacity-100 transition-opacity text-xs"
              >
                Delete
              </button>
            </div>
            <div className="p-3 space-y-1">
              <p
                className="text-xs text-zinc-500 truncate"
                title={img.prompt}
              >
                {img.prompt || "No prompt recorded"}
              </p>
              <p className="text-xs text-zinc-600">
                {new Date(img.timestamp).toLocaleDateString()}{" "}
                {new Date(img.timestamp).toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
