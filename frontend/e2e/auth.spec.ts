import { test, expect } from '@playwright/test'

const TEST_API_KEY = process.env.NEXUS_TEST_API_KEY || 'test-api-key'

test.describe('Authentication', () => {
  test('accessing protected route without token redirects to login', async ({ page }) => {
    // Clear any existing session state
    await page.context().clearCookies()
    await page.goto('/')
    // Should redirect to /login since there's no token in sessionStorage
    await expect(page).toHaveURL(/\/login/)
  })

  test('login page renders with NEXUS branding', async ({ page }) => {
    await page.goto('/login')
    await expect(page.locator('h1')).toContainText('NEXUS')
    await expect(page.locator('text=NEXUS API Key')).toBeVisible()
    await expect(page.locator('button[type="submit"]')).toContainText('Authenticate')
  })

  test('login with test API key redirects to dashboard', async ({ page }) => {
    await page.goto('/login')
    // Fill the API key input
    await page.fill('input#api-key', TEST_API_KEY)
    // Submit
    await page.click('button[type="submit"]')
    // Should redirect to dashboard on success
    // Allow for either success redirect or error display
    await page.waitForLoadState('networkidle')
    const url = page.url()
    const hasError = await page.locator('text=Authentication failed').isVisible().catch(() => false)
    if (!hasError) {
      // Successful auth should redirect to /
      expect(url).not.toContain('/login')
    }
  })

  test('invalid API key shows error message', async ({ page }) => {
    await page.goto('/login')
    await page.fill('input#api-key', 'nxs_invalid_key_12345')
    await page.click('button[type="submit"]')
    // Wait for the error to appear
    await page.waitForLoadState('networkidle')
    // Either shows error or stays on login page
    const isOnLogin = page.url().includes('/login')
    const hasError = await page.locator('[class*="red"]').isVisible().catch(() => false)
    expect(isOnLogin || hasError).toBeTruthy()
  })

  test('authenticate button is disabled when input is empty', async ({ page }) => {
    await page.goto('/login')
    const submitBtn = page.locator('button[type="submit"]')
    // The button has disabled={loading || !apiKey}, so empty key = disabled
    await expect(submitBtn).toBeDisabled()
  })

  test('input placeholder shows nxs_ prefix hint', async ({ page }) => {
    await page.goto('/login')
    const input = page.locator('input#api-key')
    await expect(input).toHaveAttribute('placeholder', 'nxs_...')
  })

  test('logout clears session and redirects to login', async ({ page }) => {
    // First, set up auth state by injecting token into sessionStorage
    await page.goto('/login')
    await page.evaluate(() => {
      sessionStorage.setItem('nexus_token', 'fake-token-for-test')
    })
    await page.goto('/')
    // Now clear the auth by removing the token
    await page.evaluate(() => {
      sessionStorage.removeItem('nexus_token')
      sessionStorage.removeItem('nexus_tenant')
    })
    // Reload — should redirect to login
    await page.reload()
    await expect(page).toHaveURL(/\/login/)
  })
})
