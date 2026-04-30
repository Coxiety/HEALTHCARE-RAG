package com.webdinhduong.chatbot.util;

import java.security.Key;
import java.nio.charset.StandardCharsets;
import java.util.Date;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.MalformedJwtException;
import io.jsonwebtoken.UnsupportedJwtException;
import io.jsonwebtoken.security.Keys;
import io.jsonwebtoken.security.SignatureException;

@Component
public class JwtUtil {
    // Dùng secret cố định từ config để token không bị invalid sau mỗi lần restart app.
    private final Key key;
    private final long expirationTime = 604800000L; // 7 ngày (tính bằng miliseconds)

    public JwtUtil(
            @Value("${jwt.secret:chatbot-default-secret-key-please-change-this-in-production-2026}") String jwtSecret) {
        this.key = Keys.hmacShaKeyFor(jwtSecret.getBytes(StandardCharsets.UTF_8));
    }

    // --- 1. Hàm tạo Token (Dùng trong AuthController khi Login) ---
    public String generateToken(String username) {
        return Jwts.builder()
                .setSubject(username)
                .setIssuedAt(new Date())
                .setExpiration(new Date(System.currentTimeMillis() + expirationTime))
                .signWith(key)
                .compact();
    }

    // --- 2. Hàm giải mã lấy Username (Dùng trong JwtFilter) ---
    public String extractUsername(String token) {
        return Jwts.parserBuilder()
                .setSigningKey(key)
                .build()
                .parseClaimsJws(token)
                .getBody()
                .getSubject();
    }

    // --- 3. Hàm KIỂM TRA Token (Dùng trong JwtFilter) ---
    public boolean validateToken(String token) {
        try {
            Jwts.parserBuilder()
                    .setSigningKey(key)
                    .build()
                    .parseClaimsJws(token);
            return true; // Token hoàn toàn hợp lệ
        } catch (SignatureException e) {
            System.out.println("Lỗi: Chữ ký JWT không khớp!");
        } catch (MalformedJwtException e) {
            System.out.println("Lỗi: Định dạng Token không đúng!");
        } catch (ExpiredJwtException e) {
            System.out.println("Lỗi: Token đã hết hạn sử dụng!");
        } catch (UnsupportedJwtException e) {
            System.out.println("Lỗi: Token không được hỗ trợ!");
        } catch (IllegalArgumentException e) {
            System.out.println("Lỗi: Chuỗi Token bị trống!");
        } catch (Exception e) {
            System.out.println("Lỗi xác thực: " + e.getMessage());
        }
        return false; // Nếu rơi vào bất kỳ lỗi nào ở trên
    }
}