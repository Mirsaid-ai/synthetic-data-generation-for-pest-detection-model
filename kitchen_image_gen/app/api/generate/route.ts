import { NextRequest, NextResponse } from "next/server";
import { getGeminiClient, KITCHEN_PROMPT_PREFIX } from "@/lib/gemini";

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();
    if (!prompt || typeof prompt !== "string") {
      return NextResponse.json(
        { error: "prompt is required" },
        { status: 400 }
      );
    }

    const genAI = getGeminiClient();
    const model = genAI.getGenerativeModel({
      model: "gemini-2.0-flash-exp",
      generationConfig: {
        // @ts-expect-error -- image generation response modality
        responseModalities: ["image", "text"],
      },
    });

    const fullPrompt = KITCHEN_PROMPT_PREFIX + prompt;
    const result = await model.generateContent(fullPrompt);
    const response = result.response;

    const parts = response.candidates?.[0]?.content?.parts;
    if (!parts) {
      return NextResponse.json(
        { error: "No response from Gemini" },
        { status: 502 }
      );
    }

    for (const part of parts) {
      if (part.inlineData) {
        const base64 = part.inlineData.data;
        const mimeType = part.inlineData.mimeType || "image/png";
        return NextResponse.json({
          image: `data:${mimeType};base64,${base64}`,
          prompt: fullPrompt,
        });
      }
    }

    const textParts = parts
      .filter((p: any) => p.text)
      .map((p: any) => p.text)
      .join("\n");
    return NextResponse.json(
      { error: "No image generated", text: textParts },
      { status: 422 }
    );
  } catch (err: any) {
    console.error("Generate error:", err);
    return NextResponse.json(
      { error: err.message || "Generation failed" },
      { status: 500 }
    );
  }
}
