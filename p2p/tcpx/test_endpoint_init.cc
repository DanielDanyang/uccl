#include "tcpx_endpoint.h"
#include <iostream>
#include <cstdlib>

int main() {
    std::cout << "=== TCPX Endpoint Initialization Test ===" << std::endl;
    
    // Set debug mode for more verbose output
    setenv("UCCL_TCPX_DEBUG", "1", 1);
    
    try {
        std::cout << "\n[Test 1] Creating Endpoint with GPU 0, 4 CPUs..." << std::endl;
        
        // Test basic endpoint creation
        Endpoint* endpoint = new Endpoint(0, 4);
        
        std::cout << "✓ Endpoint created successfully!" << std::endl;
        
        std::cout << "\n[Test 2] Testing endpoint destruction..." << std::endl;
        delete endpoint;
        std::cout << "✓ Endpoint destroyed successfully!" << std::endl;
        
        std::cout << "\n=== Endpoint initialization test completed successfully! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Endpoint initialization failed with exception: " << e.what() << std::endl;
        std::cout << "\nPossible issues:" << std::endl;
        std::cout << "  - TCPX plugin not found or failed to load" << std::endl;
        std::cout << "  - No TCPX devices available" << std::endl;
        std::cout << "  - GPU runtime initialization failed" << std::endl;
        std::cout << "\nTroubleshooting:" << std::endl;
        std::cout << "  - Check UCCL_TCPX_PLUGIN_PATH environment variable" << std::endl;
        std::cout << "  - Ensure TCPX plugin is installed" << std::endl;
        std::cout << "  - Verify GPU devices are available" << std::endl;
        return 1;
    }
}
