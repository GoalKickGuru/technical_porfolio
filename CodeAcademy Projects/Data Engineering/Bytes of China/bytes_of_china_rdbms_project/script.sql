-- =============================================================================
-- Bytes of China — script.sql
-- Complete the TODOs in order (Tasks 1–18).
-- After Task 8, append projectdata.sql (or \i projectdata.sql in psql).
-- Do not peek at the cheat sheet until you are stuck for more than 15 minutes.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Task Group 1 — Create tables and primary keys
-- -----------------------------------------------------------------------------

-- TASK 1–2. restaurant and address
-- restaurant holds the venue facts from the website blurb.
-- address holds the street / city / map link.
-- You will connect them with a 1:1 FK in Task 6 — not yet.
CREATE TABLE restaurant (
  -- TODO: columns + PRIMARY KEY that match projectdata.sql INSERT shape
);

CREATE TABLE address (
  -- TODO: columns + PRIMARY KEY
);

-- TASK 2. Validate PKs (keep these queries; they should return one row each)
-- SELECT constraint_name, table_name
-- FROM information_schema.table_constraints
-- WHERE table_name IN ('restaurant', 'address')
--   AND constraint_type = 'PRIMARY KEY';

-- TASK 3. category
-- id is a 2-character code: 'C', 'LS', 'HS', ...
-- description is NULL for most categories; Luncheon Specials stores hours / sides.
CREATE TABLE category (
  -- TODO
);

-- TASK 4. dish
-- spicy flag is boolean. Price does NOT live here (see Task 8).
CREATE TABLE dish (
  -- TODO
);

-- TASK 5. review
-- One restaurant, many reviews. FK comes in Task 7.
CREATE TABLE review (
  -- TODO
);

-- -----------------------------------------------------------------------------
-- Task Group 2 — Relationships and foreign keys
-- -----------------------------------------------------------------------------

-- TASK 6. 1:1  restaurant ↔ address
-- Add the FK on the dependent table. The sample INSERT for address ends with 1
-- (that value is restaurant_id). Make the FK UNIQUE so it stays 1:1.

-- TASK 7. 1:N  restaurant → review
-- Add restaurant_id on review (sample INSERT already includes it as the last value).

-- TASK 8. M:N  category ↔ dish via categories_dishes
-- Composite PK (category_id, dish_id) — required by this course.
-- price money lives HERE because the same dish can cost $6.95 as Chicken
-- and $8.95 as a Luncheon Special.
CREATE TABLE categories_dishes (
  -- TODO
);

-- Validate FKs / PKs with information_schema before you load data.

-- -----------------------------------------------------------------------------
-- Task Group 3 — Load sample data
-- -----------------------------------------------------------------------------
-- TASK 9. After the schema matches the INSERT column order:
--   \i projectdata.sql
-- or paste the file at the end of this script.

-- -----------------------------------------------------------------------------
-- Task Group 4 — Queries
-- -----------------------------------------------------------------------------

-- TASK 10. Restaurant name, street number + street name, telephone

-- TASK 11. Best rating as best_rating

-- TASK 12. dish name, price, category — sorted by dish name (8 rows)

-- TASK 13. Same columns — sorted by category name

-- TASK 14. Spicy dishes, price, category (3 rows)

-- TASK 15. dish_id and COUNT(dish_id) AS dish_count from categories_dishes
--          GROUP BY dish_id

-- TASK 16. Only dishes that appear more than once (HAVING)

-- TASK 17. dish_name + dish_count for those multi-category dishes

-- TASK 18. best_rating AND the review description (subquery on MAX(rating))
