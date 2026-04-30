package com.webdinhduong.chatbot.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Entity
@Table(name = "chat_messages")
@Data
public class ChatMessage {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;

    private String role; // "user" | "ai"

    @Column(columnDefinition = "TEXT")
    private String content;

    private String intent; // NUTRITION_LOOKUP | HEALTH_ADVICE | BOTH | GREETING

    @Column(columnDefinition = "TEXT")
    private String entitiesJson; // JSON string của entities map

    @Column(columnDefinition = "TEXT")
    private String sourcesJson; // JSON string của sources list

    private String energyAmount; // amountPer100g dưới dạng string
    private String energyUnit;   // "kcal", "g", ...

    private String sessionId; // UUID của cuộc hội thoại

    private LocalDateTime createdAt = LocalDateTime.now();
}
