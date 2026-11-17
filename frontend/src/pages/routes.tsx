import { LibraryPage } from "@/features/library/LibraryPage";
import { EditorPage } from "@/features/editor/EditorPage";
import { VideosPage } from "@/pages/VideosPage";

export const routes = [
  { path: "/videos", element: <VideosPage /> },
  { path: "/clips", element: <LibraryPage /> },
  { path: "/clips/:clipId", element: <EditorPage /> },
  { path: "*", element: <VideosPage /> },
];
