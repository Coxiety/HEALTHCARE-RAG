package com.webdinhduong.chatbot.dto;

import java.util.List;
import java.util.Map;

public record ChatResponse(
        String answer,
        String intent,
        Map<String, List<String>> entities,
        List<String> sources,
        EnergyInfo energy
) {
    public record EnergyInfo(Object amountPer100g, String unitName) {}
}
