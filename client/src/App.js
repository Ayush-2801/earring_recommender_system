import { useState, useRef } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState();
  const fileUploadRef = useRef();
  console.log("file", file);
  // const handleSubmit=async (e)=>{
  //   e.preventDefault();
  //   // const formData = new FormData();
  //   // const uploadedFile = fileUploadRef.current.files[0];
  //   // console.log("fileRef", uploadedFile);
  //   // formData.append("file", uploadedFile);
  //   // console.log("event", formData);
  //   try {
  //     const message = await axios.post('http://localhost:5000/pred', formData, { headers: {'Content-Type': 'multipart/form-data'}})
  //     if(message){
  //         alert("File uploaded");
  //     }else{
  //         alert("Try again later");
  //     }
  //     // console.log("done", name, email, password);
  //   } catch (error) {
  //       console.log("error");
  //   }
  // }
  const handleSubmit = async (event)=>{
    event.preventDefault();
    const formData = new FormData();
    formData.append('file', file); // Append the file to formData with a key 'image'

    try {
      const response = await fetch('http://localhost:5000/pred', {
        method: 'POST',
        body: formData,
      });
      console.log(response);
      const result = await response.json();
      console.log(result); // Handle the result as needed
    } catch (error) {
      console.error('Error:', error);
    }
  }
  return (
        <div className="container mt-5">
            <h2 className="mb-4">Access Webcam or Upload File</h2>
            <div className="row">
              <div className="col-md-6">
                  <button className="btn btn-primary btn-block">Access Webcam</button>
                  <video id="webcam" className="mt-3" width="100%" height="auto"></video>
              </div>
              <div className="col-6">
                <form onSubmit={handleSubmit} encType='multipart/form-data'>
                  <input type="file" className="form-control-file mt-3" name="file" accept="image/*" onChange={(event)=>{
                    console.log(event.target.files[0]);
                    setFile(event.target.files[0]);
                  }} ref={fileUploadRef}/>
                  <button className="btn btn-primary btn-block mt-3" type="submit">Upload File</button>
                  <p id="errormsg" className="text-danger"></p>
                </form>
              </div>
            </div>
        </div>


        
  
  );
}

export default App;
