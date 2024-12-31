import express from 'express';
import request from 'request';
import path from 'path';
import axios from 'axios';
import cors from 'cors';
import { extname } from 'path';
import FormData from 'form-data';
import { createReadStream, unlinkSync, appendFile } from 'fs';
import multer from 'multer';

const app = express();
app.use(cors({
    origin: "http://192.168.3.198:3001",
    credentials: true
}));

const storage = multer.diskStorage({
    destination: path.join('./uploads'),
    filename: (req, file, cb) => {
        cb(null, Date.now() + extname(file.originalname));
    }
});

var upload = multer({ storage: storage });
var type = upload.single('file');
const PORT = 5000;

app.get('/', function(req, res) {
    console.log(req);
    request('http://192.168.3.199:1234/', function (error, response, body) {
        console.error('error:', error);
        console.log('statusCode:', response && response.statusCode);
        console.log('body:', body);
        res.send(body);
    });
});

// const file = path.resolve("./Heart0.jpg")
app.post('/pred', type, async function(req, res){
    try {
        const imageFilePath = req.file.path;
        const formData = new FormData();
        formData.append('file', createReadStream(imageFilePath));
    
        // Send the image to Flask
        const flaskResponse = await axios.post('http://192.168.3.199:1234/predict', formData, {
          headers: {
            ...formData.getHeaders(),
          },
        });

        // Delete the local copy of the file (optional cleanup)
        unlinkSync(imageFilePath);
        console.log("flask response", flaskResponse.data)

        // Send Flask's response back to React
        res.json(flaskResponse.data);
      } catch (error) {
        console.error('Error:', error);
        res.status(500).json({ message: 'Server Error' });
      }
    // console.log("req", req);
    // const file = req.body.file;
    // console.log(file);
    // if (!file) {
    //     return res.status(400).send({"data":'No file uploaded.'});
    // }
    // const form = new FormData();
    // form.append('file', createReadStream(file.path), file.originalname);
    // try {
    //     const response = request.post('http://192.168.3.199:1234/predict', form, {
    //         headers: {
    //             ...form.getHeaders(),
    //         },
    //     });
    //     console.log("response",response.data);
    //     res.json(response.data);
    // } catch (error) {
    //     console.error('Error uploading file to Flask server:', error);
    //     res.status(500).send({"data":'Failed to upload file to Flask server.'});
    // } finally {
    //     unlinkSync(file.path);
    // }
    // try {
    //     console.log(req.file);
    //     const file = req.file;  // This contains the uploaded file information
    //     // Extract file information
    //     const { originalname, filename, path, mimetype, size } = file;
    //     // const file = req.file.path;
    //     // console.log(await axios.post("http://192.168.3.199:5000/predict", req.file.path))
    //     // const response = await axios.post("http://192.168.3.199:5000/predict", file);
    //     // console.log("response", response);
    // } catch (error) {
    //     console.log(error);
    // }
})

app.listen(PORT, function (){
    console.log(`Listening on Port ${PORT}`);
});