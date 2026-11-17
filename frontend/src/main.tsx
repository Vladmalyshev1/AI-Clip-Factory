import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/AppShell";
import { ToastViewport } from "@/components/ToastViewport";
import { routes } from "@/pages/routes";
import "./index.css";

const root = document.getElementById("root");
if (root === null) throw new Error("#root is missing from index.html");

createRoot(root).render(
  <StrictMode>
    <BrowserRouter>
      <AppShell>
        <Routes>
          {routes.map((route) => (
            <Route key={route.path} path={route.path} element={route.element} />
          ))}
        </Routes>
      </AppShell>
      <ToastViewport />
    </BrowserRouter>
  </StrictMode>,
);
