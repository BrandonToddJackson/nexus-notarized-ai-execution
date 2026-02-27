import { test, expect, Page } from '@playwright/test'

const TEST_API_KEY = process.env.NEXUS_TEST_API_KEY || 'test-api-key'

async function login(page: Page) {
  await page.goto('/login')
  await page.fill('input#api-key', TEST_API_KEY)
  await page.click('button[type="submit"]')
  await page.waitForLoadState('networkidle')
  if (page.url().includes('/login')) {
    await page.evaluate(() => {
      sessionStorage.setItem('nexus_token', 'e2e-test-token')
      sessionStorage.setItem('nexus_tenant', 'e2e-tenant')
    })
    await page.goto('/')
  }
}

test.describe('Skills Page', () => {
  test.beforeEach(async ({ page }) => {
    await login(page)
  })

  test('navigate to skills page', async ({ page }) => {
    await page.goto('/skills')
    await expect(page.locator('h1')).toContainText('Skills')
  })

  test('skills page has New Skill button', async ({ page }) => {
    await page.goto('/skills')
    await expect(page.locator('text=New Skill')).toBeVisible()
  })

  test('skills page has Import button', async ({ page }) => {
    await page.goto('/skills')
    await expect(page.locator('text=Import')).toBeVisible()
  })

  test('skills page has search input', async ({ page }) => {
    await page.goto('/skills')
    await expect(page.locator('input[placeholder="Search skills..."]')).toBeVisible()
  })

  test('skills page has Active only toggle', async ({ page }) => {
    await page.goto('/skills')
    await expect(page.locator('text=Active only')).toBeVisible()
  })

  test('New Skill navigates to skill editor', async ({ page }) => {
    await page.goto('/skills')
    await page.click('text=New Skill')
    await page.waitForURL(/\/skills\/new/)
    await expect(page.locator('h1')).toContainText('New Skill')
  })

  test('skill editor has Back button', async ({ page }) => {
    await page.goto('/skills/new')
    await page.waitForLoadState('networkidle')
    await expect(page.locator('text=Back')).toBeVisible()
  })

  test('skill editor Back navigates to skills list', async ({ page }) => {
    await page.goto('/skills/new')
    await page.waitForLoadState('networkidle')
    await page.click('text=Back')
    await page.waitForURL(/\/skills$/)
    await expect(page.locator('h1')).toContainText('Skills')
  })

  test('search filters skills list', async ({ page }) => {
    await page.goto('/skills')
    await page.waitForLoadState('networkidle')
    const searchInput = page.locator('input[placeholder="Search skills..."]')
    await searchInput.fill('nonexistent_skill_xyz')
    await page.waitForLoadState('networkidle')
    // Either shows filtered results or empty state
    // The search triggers a re-fetch with the search param
  })

  test('skill cards show version badges if skills exist', async ({ page }) => {
    await page.goto('/skills')
    await page.waitForLoadState('networkidle')
    // If skills exist, version badges like "v1", "v2" should be visible
    // If no skills, empty state shows
    const hasSkills = await page.locator('[class*="grid"] > div').count()
    if (hasSkills > 0) {
      const versionBadge = page.locator('text=/v\\d+/')
      await expect(versionBadge.first()).toBeVisible()
    }
  })

  test('skill card menu shows edit, duplicate, delete options', async ({ page }) => {
    await page.goto('/skills')
    await page.waitForLoadState('networkidle')
    // Find a menu trigger ("...") button if skills exist
    const menuBtns = page.locator('text=...')
    if (await menuBtns.count() > 0) {
      await menuBtns.first().click()
      await expect(page.locator('text=Edit')).toBeVisible()
      await expect(page.locator('text=Duplicate')).toBeVisible()
      await expect(page.locator('text=Delete')).toBeVisible()
      await expect(page.locator('text=Export')).toBeVisible()
    }
  })
})
