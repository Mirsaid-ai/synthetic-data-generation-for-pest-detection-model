import { NextRequest, NextResponse } from "next/server";
import { listImages, deleteImage } from "@/lib/storage";

export async function GET() {
  const images = listImages();
  return NextResponse.json({ images });
}

export async function DELETE(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const id = searchParams.get("id");
  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }
  const deleted = deleteImage(id);
  if (!deleted) {
    return NextResponse.json({ error: "Image not found" }, { status: 404 });
  }
  return NextResponse.json({ success: true });
}
