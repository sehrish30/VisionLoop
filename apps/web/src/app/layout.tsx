import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisionLoop — Clothing classification studio",
  description: "A local workspace for classifying clothing, reviewing labels, and training better models.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
