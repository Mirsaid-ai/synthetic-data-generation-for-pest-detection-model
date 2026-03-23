import ImageGenerator from "@/components/ImageGenerator";

export default function GeneratePage() {
  return (
    <div>
      <div className="mb-8">
        <h2 className="text-2xl font-bold">Generate Kitchen Images</h2>
        <p className="text-zinc-500 mt-1">
          Use Gemini to create diverse kitchen backgrounds for pest detection
          training data.
        </p>
      </div>
      <ImageGenerator />
    </div>
  );
}
