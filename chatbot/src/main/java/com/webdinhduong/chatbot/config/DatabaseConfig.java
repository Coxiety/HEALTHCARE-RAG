package com.webdinhduong.chatbot.config;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.sql.DataSource;

import org.springframework.boot.jdbc.DataSourceBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class DatabaseConfig {

    private static final String DATABASE_NAME = "dinhduongdb";

    @Bean
    public DataSource dataSource() {
        Path databaseBasePath = resolveDatabaseBasePath();
        String jdbcUrl = "jdbc:h2:file:" + databaseBasePath.toString().replace('\\', '/') + ";AUTO_SERVER=TRUE";

        System.out.println("[Chatbot DB] Using H2 database at: " + databaseBasePath);

        return DataSourceBuilder.create()
                .driverClassName("org.h2.Driver")
                .url(jdbcUrl)
                .username("sa")
                .password("")
                .build();
    }

    private Path resolveDatabaseBasePath() {
        Path moduleRoot = resolveModuleRoot();
        Path canonicalBasePath = moduleRoot.resolve("data").resolve(DATABASE_NAME);
        Path legacyBasePath = moduleRoot.getParent().resolve("data").resolve(DATABASE_NAME);

        Path selectedBasePath = chooseNewestExistingBasePath(canonicalBasePath, legacyBasePath);
        createParentDirectory(selectedBasePath);
        return selectedBasePath;
    }

    private Path resolveModuleRoot() {
        try {
            Path location = Paths.get(DatabaseConfig.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                    .toAbsolutePath();
            Path targetDir = location.getParent();
            if (targetDir == null || targetDir.getParent() == null) {
                throw new IllegalStateException("Cannot resolve module root from: " + location);
            }
            return targetDir.getParent();
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Cannot resolve module root for database path.", e);
        }
    }

    private Path chooseNewestExistingBasePath(Path canonicalBasePath, Path legacyBasePath) {
        Path canonicalDbFile = basePathToDbFile(canonicalBasePath);
        Path legacyDbFile = basePathToDbFile(legacyBasePath);

        boolean canonicalExists = Files.exists(canonicalDbFile);
        boolean legacyExists = Files.exists(legacyDbFile);

        if (canonicalExists && legacyExists) {
            try {
                return Files.getLastModifiedTime(canonicalDbFile).compareTo(Files.getLastModifiedTime(legacyDbFile)) >= 0
                        ? canonicalBasePath
                        : legacyBasePath;
            } catch (IOException e) {
                return canonicalBasePath;
            }
        }

        if (canonicalExists) {
            return canonicalBasePath;
        }

        if (legacyExists) {
            return legacyBasePath;
        }

        return canonicalBasePath;
    }

    private Path basePathToDbFile(Path basePath) {
        return Path.of(basePath.toString() + ".mv.db");
    }

    private void createParentDirectory(Path databaseBasePath) {
        try {
            Path parent = databaseBasePath.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Cannot create database directory.", e);
        }
    }
}