#include <gtest/gtest.h>

extern "C" {
    int configure_engine(const char* json_str);
    void free_engine();
}

TEST(AiEngineTest, ConfigureEngine_ValidJson) {
    const char* valid_json = R"({"n_threads": 8, "n_ctx": 4096, "n_batch": 1024, "img_max_tokens": 256})";

    EXPECT_EQ(configure_engine(valid_json), 0);
}

TEST(AiEngineTest, ConfigureEngine_PartialJson) {
    const char* partial_json = R"({"n_threads": 2})";
    EXPECT_EQ(configure_engine(partial_json), 0);
}

TEST(AiEngineTest, ConfigureEngine_InvalidJson) {
    const char* invalid_json = "this is not json";

    EXPECT_EQ(configure_engine(invalid_json), -1);
}
