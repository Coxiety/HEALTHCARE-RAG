package com.webdinhduong.chatbot.controller;

import com.webdinhduong.chatbot.dto.ChatResponse;
import com.webdinhduong.chatbot.entity.ChatMessage;
import com.webdinhduong.chatbot.entity.FoodRecord;
import com.webdinhduong.chatbot.entity.User;
import com.webdinhduong.chatbot.repository.ChatMessageRepository;
import com.webdinhduong.chatbot.repository.FoodRecordRepository;
import com.webdinhduong.chatbot.repository.UserRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/chat")
@CrossOrigin(origins = "*")
public class ChatController {
    private static final Logger log = LoggerFactory.getLogger(ChatController.class);

    @Autowired
    private FoodRecordRepository foodRecordRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private ChatMessageRepository chatMessageRepository;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${rag.apiUrl:http://localhost:8000}")
    private String ragApiUrl;

    @Value("${rag.timeoutMs:240000}")
    private int ragTimeoutMs;

    // API nhận tin nhắn từ người dùng
    @PostMapping("/send")
    public ResponseEntity<?> sendMessage(@RequestParam Long userId, @RequestBody Map<String, String> payload) {
        String userMessage = payload.get("message");
        String sessionId   = payload.get("sessionId");

        User user = userRepository.findById(userId).orElse(null);
        if (user != null) {
            ChatMessage userMsg = new ChatMessage();
            userMsg.setUser(user);
            userMsg.setRole("user");
            userMsg.setContent(userMessage);
            userMsg.setSessionId(sessionId);
            chatMessageRepository.save(userMsg);
        }

        String aiResponse = null;
        String intent = null;
        Map<String, List<String>> entities = null;
        List<String> sources = new ArrayList<>();
        ChatResponse.EnergyInfo energyDto = null;
        String savedFoodName = null;
        String savedCalories = null;

        try {
            Map<String, Object> ragResult = callRagModel(userMessage);

            aiResponse = (String) ragResult.getOrDefault("answer", null);
            intent = (String) ragResult.getOrDefault("intent", null);

            @SuppressWarnings("unchecked")
            Map<String, Object> rawEntities = (Map<String, Object>) ragResult.get("entities");
            if (rawEntities != null) {
                entities = new java.util.LinkedHashMap<>();
                for (Map.Entry<String, Object> entry : rawEntities.entrySet()) {
                    @SuppressWarnings("unchecked")
                    List<String> vals = (List<String>) entry.getValue();
                    entities.put(entry.getKey(), vals != null ? vals : new ArrayList<>());
                }
                List<String> foods = entities.get("foods");
                if (foods != null && !foods.isEmpty()) {
                    savedFoodName = foods.get(0);
                }
            }

            @SuppressWarnings("unchecked")
            List<String> rawSources = (List<String>) ragResult.get("sources");
            if (rawSources != null) sources = rawSources;

            @SuppressWarnings("unchecked")
            Map<String, Object> energy = (Map<String, Object>) ragResult.get("energy");
            if (energy != null) {
                Object amount = energy.get("amountPer100g");
                Object unit = energy.get("unitName");
                if (amount != null && unit != null) {
                    energyDto = new ChatResponse.EnergyInfo(amount, (String) unit);
                    savedCalories = amount + " " + unit + " / 100g";
                }
            }

        } catch (ResourceAccessException e) {
            log.error("Không kết nối được RAG server tại {}", ragApiUrl, e);
            aiResponse = "Hệ thống AI chưa khởi động. Hãy chạy lệnh: python main/rag_server.py";
        } catch (Exception e) {
            log.error("Lỗi khi gọi RAG server", e);
            aiResponse = "Hệ thống AI gặp lỗi. Vui lòng thử lại sau.";
        }

        if (aiResponse == null) {
            aiResponse = "Xin lỗi, tôi chưa hiểu ý bạn.";
        }

        if (savedFoodName != null && user != null) {
            FoodRecord record = new FoodRecord();
            record.setUser(user);
            record.setFoodName(savedFoodName);
            record.setCalories(savedCalories != null ? savedCalories : "Đang cập nhật từ AI...");
            foodRecordRepository.save(record);
        }

        if (user != null) {
            ChatMessage aiMsg = new ChatMessage();
            aiMsg.setUser(user);
            aiMsg.setRole("ai");
            aiMsg.setContent(aiResponse);
            aiMsg.setIntent(intent);
            aiMsg.setSessionId(sessionId);
            try {
                if (entities != null) aiMsg.setEntitiesJson(objectMapper.writeValueAsString(entities));
                if (!sources.isEmpty()) aiMsg.setSourcesJson(objectMapper.writeValueAsString(sources));
            } catch (Exception e) {
                log.warn("Không serialize được entities/sources", e);
            }
            if (energyDto != null) {
                aiMsg.setEnergyAmount(String.valueOf(energyDto.amountPer100g()));
                aiMsg.setEnergyUnit(energyDto.unitName());
            }
            chatMessageRepository.save(aiMsg);
        }

        return ResponseEntity.ok(new ChatResponse(aiResponse, intent, entities, sources, energyDto));
    }

    // Danh sách sessions của user (dùng cho sidebar)
    @GetMapping("/sessions/{userId}")
    public ResponseEntity<?> getSessions(@PathVariable Long userId) {
        List<ChatMessage> firstMsgs = chatMessageRepository.findFirstUserMessagePerSession(userId);
        List<Map<String, Object>> result = firstMsgs.stream().map(m -> {
            String title = m.getContent();
            if (title != null && title.length() > 50) title = title.substring(0, 50) + "...";
            Map<String, Object> entry = new java.util.HashMap<>();
            entry.put("sessionId", m.getSessionId());
            entry.put("title",     title != null ? title : "(Cuộc hội thoại)");
            entry.put("createdAt", m.getCreatedAt().toString());
            return entry;
        }).collect(Collectors.toList());
        return ResponseEntity.ok(result);
    }

    // Tải toàn bộ messages của 1 session
    @GetMapping("/session/{sessionId}")
    public ResponseEntity<List<ChatMessage>> getSession(@PathVariable String sessionId) {
        return ResponseEntity.ok(chatMessageRepository.findBySessionIdOrderByCreatedAtAsc(sessionId));
    }

    // API lịch sử chat (50 tin nhắn gần nhất, mọi session)
    @GetMapping("/history/{userId}")
    public ResponseEntity<List<ChatMessage>> getHistory(@PathVariable Long userId) {
        List<ChatMessage> msgs = chatMessageRepository.findTop50ByUserIdOrderByCreatedAtDesc(userId);
        java.util.Collections.reverse(msgs);
        return ResponseEntity.ok(msgs);
    }

    // API Dashboard
    @GetMapping("/dashboard/{userId}")
    public ResponseEntity<List<FoodRecord>> getDashboard(@PathVariable Long userId) {
        return ResponseEntity.ok(foodRecordRepository.findByUserIdOrderByCreatedAtDesc(userId));
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> callRagModel(String userMessage) {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(10_000);
        factory.setReadTimeout(ragTimeoutMs);

        RestTemplate restTemplate = new RestTemplate(factory);
        ResponseEntity<Map> response = restTemplate.postForEntity(
                ragApiUrl + "/ask",
                Map.of("message", userMessage),
                Map.class
        );

        if (response.getBody() == null) {
            throw new RuntimeException("RAG server trả về dữ liệu rỗng.");
        }
        return (Map<String, Object>) response.getBody();
    }
}
