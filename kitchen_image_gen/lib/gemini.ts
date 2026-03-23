import { GoogleGenerativeAI } from "@google/generative-ai";

const API_KEY = process.env.GEMINI_API_KEY;

if (!API_KEY) {
  console.warn(
    "GEMINI_API_KEY not set. Create a .env.local file with your key."
  );
}

export function getGeminiClient() {
  if (!API_KEY) {
    throw new Error("GEMINI_API_KEY environment variable is not set");
  }
  return new GoogleGenerativeAI(API_KEY);
}

export const KITCHEN_PROMPT_PREFIX =
  "Generate a photorealistic image of a commercial kitchen interior. " +
  "The image should show the floor clearly with good perspective. " +
  "No people or animals should be present. ";
