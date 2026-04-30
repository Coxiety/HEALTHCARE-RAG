package com.webdinhduong.chatbot.repository;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import com.webdinhduong.chatbot.entity.FoodRecord;

public interface FoodRecordRepository extends JpaRepository<FoodRecord, Long> {
    List<FoodRecord> findByUserIdOrderByCreatedAtDesc(Long userId);
}