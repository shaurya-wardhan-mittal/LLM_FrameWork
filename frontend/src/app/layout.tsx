import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Automatic Fine-Tune Framework",
  description: "Upload a dataset and let the framework prepare, train, evaluate, and export.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
