# InfraNova AI Frontend

This is the React frontend for InfraNova AI, built with Vite. It provides a polished, responsive, dark-themed "glassmorphism" interface for colorizing Landsat 9 thermal infrared satellite imagery.

## Features

- **Interactive Comparison Slider:** Drag to smoothly compare the raw thermal input with the generated RGB output.
- **CLAHE Post-Processing:** Built-in toggle to apply Contrast Limited Adaptive Histogram Equalization in LAB color space for enhanced visibility.
- **Test-Time Augmentation (TTA):** Toggle advanced 4-way geometric ensembling for higher quality (at the cost of longer inference time).
- **Responsive 100vh Layout:** Designed to act like a desktop application, fitting perfectly within the browser viewport without scrolling.
- **API Health Indicator:** Real-time status dot indicating backend connectivity.

## Tech Stack

- **Framework:** React 18 + Vite
- **Styling:** Vanilla CSS (custom design system and tokens in `index.css`)
- **No Heavy CSS Frameworks:** Strict adherence to pure CSS to keep the application lightning-fast and perfectly tailored to the project's unique aesthetic.

## Development

The frontend expects the FastAPI backend to be running (usually on `http://localhost:8000`).

To start the Vite development server:

```bash
npm install
npm run dev
```

You can then view the app at [http://localhost:5173](http://localhost:5173).

## Production Build

In production, the frontend is built into static files and served directly by the FastAPI backend to eliminate CORS issues and simplify deployment.

```bash
npm run build
```

This will output the static assets to the `dist/` directory, which FastAPI is configured to mount at the root path (`/`).
