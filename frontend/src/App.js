import React, { useState } from "react";
import Home from "./pages/Home";
import Analysis from "./pages/Analysis";

function App() {
  const [uploadInfo, setUploadInfo] = useState(null);

  return (
    <div>
      {!uploadInfo ? (
        <Home onUploadSuccess={setUploadInfo} />
      ) : (
        <Analysis uploadInfo={uploadInfo} />
      )}
    </div>
  );
}

export default App;