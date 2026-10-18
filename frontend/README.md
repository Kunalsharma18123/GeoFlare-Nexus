# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

## GeoFlare risk alerts

Open **Alerts** in the header to see all active `MEDIUM` and `HIGH` risk events,
with high-risk alerts shown first. Scroll within the alert list to browse every
event. Selecting an alert opens that event's details and dismisses the dropdown.
The dropdown also closes with Escape or an outside click and remains unavailable
while an event or investigation is open. The alert count refreshes whenever the
event feed refreshes.

The header feed indicator reads the backend's `/ingestion/status` endpoint. It
shows when a successful FIRMS ingestion last completed and warns when the last
success is more than two configured ingestion intervals old. Ingestion status
is currently in process memory and resets when the backend restarts.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.
