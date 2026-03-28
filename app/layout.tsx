import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Particles Fluid | Interactive SPH Simulation",
  description:
    "Real-time 2D fluid simulation with geometric shape formation, glow rendering, and gesture interaction.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
