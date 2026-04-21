package com.licenta.backend_ai;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.client.RestClient;

@RestController
@RequestMapping("/api/ai-detection")
@CrossOrigin(origins = "*")

public class DiagnosticController {

    private final String PYTHON_PATH = "http://127.0.0.1:8000/diagnostic/";
    private final RestClient restClient = RestClient.create();


}
