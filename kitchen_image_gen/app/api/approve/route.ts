import { NextRequest, NextResponse } from "next/server";
import { saveImage } from "@/lib/storage";
import { v4 as uuidv4 } from "uuid";

export async function POST(req: NextRequest) {
  try {
    const { image, prompt } = await req.json();
    if (!image || typeof image !== "string") {
      return NextResponse.json(
        { error: "image (base64 data URI) is required" },
        { status: 400 }
      );
    }

    const matches = image.match(/^data:image\/\w+;base64,(.+)$/);
    if (!matches) {
      return NextResponse.json(
        { error: "Invalid image data URI format" },
        { status: 400 }
      );
    }

    const buffer = Buffer.from(matches[1], "base64");
    const id = uuidv4().slice(0, 8);
    const metadata = saveImage(id, buffer, prompt || "");

    return NextResponse.json({ success: true, metadata });
  } catch (err: any) {
    console.error("Approve error:", err);
    return NextResponse.json(
      { error: err.message || "Failed to save" },
      { status: 500 }
    );
  }
}
