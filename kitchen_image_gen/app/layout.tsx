import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kitchen Image Generator",
  description: "Generate synthetic kitchen images for pest detection training",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <nav className="border-b border-zinc-800 px-6 py-4 flex items-center gap-6">
          <h1 className="text-lg font-semibold tracking-tight">
            Kitchen Image Generator
          </h1>
          <div className="flex gap-4 text-sm">
            <Link
              href="/"
              className="text-zinc-400 hover:text-white transition-colors"
            >
              Generate
            </Link>
            <Link
              href="/gallery"
              className="text-zinc-400 hover:text-white transition-colors"
            >
              Gallery
            </Link>
          </div>
        </nav>
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
