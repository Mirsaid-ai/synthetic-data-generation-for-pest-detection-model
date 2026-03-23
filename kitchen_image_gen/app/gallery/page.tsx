import ImageGallery from "@/components/ImageGallery";

export default function GalleryPage() {
  return (
    <div>
      <div className="mb-8">
        <h2 className="text-2xl font-bold">Approved Kitchen Images</h2>
        <p className="text-zinc-500 mt-1">
          These images will be used to generate synthetic pest videos.
        </p>
      </div>
      <ImageGallery />
    </div>
  );
}
