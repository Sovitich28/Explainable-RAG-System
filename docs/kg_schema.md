# Knowledge Graph Schema - EU Green Deal & Renewable Energy

## Entities (Nodes)

### Policy
- **Properties**: `name`, `description`, `year`, `source_document`, `source_chunk_id`
- **Example**: European Green Deal, Fit for 55 Package

### Target
- **Properties**: `name`, `value`, `unit`, `deadline`, `description`, `source_document`, `source_chunk_id`
- **Example**: 40% renewable energy by 2030

### Legislation
- **Properties**: `name`, `type`, `year`, `description`, `source_document`, `source_chunk_id`
- **Example**: Renewable Energy Directive III (RED III)

### Country
- **Properties**: `name`, `iso_code`, `region`
- **Example**: France (FR), Germany (DE)

### RenewableSource
- **Properties**: `name`, `type`, `description`
- **Example**: Solar, Wind, Hydroelectric

### Sector
- **Properties**: `name`, `description`
- **Example**: Transport, Industry, Buildings

## Relationships (Edges)

### MANDATES
- **From**: Policy → Target
- **Properties**: `strength`, `binding`
- **Description**: A policy mandates a specific target

### IMPLEMENTS
- **From**: Legislation → Policy
- **Properties**: `year`, `status`
- **Description**: Legislation implements a policy

### APPLIES_TO
- **From**: Target → Country
- **Properties**: `specific_value`, `deadline`
- **Description**: A target applies to a specific country

### PROMOTES
- **From**: Policy → RenewableSource
- **Properties**: `priority_level`
- **Description**: A policy promotes a renewable energy source

### IMPACTS
- **From**: Policy → Sector
- **Properties**: `impact_type`, `magnitude`
- **Description**: A policy impacts a specific sector

### SET_BY
- **From**: Target → Legislation
- **Properties**: `year`
- **Description**: A target is set by legislation

### SUPPORTS
- **From**: Country → RenewableSource
- **Properties**: `investment_level`, `capacity_mw`
- **Description**: A country supports a renewable source

## Cypher Schema Creation

```cypher
// Create constraints for unique identifiers
CREATE CONSTRAINT policy_name IF NOT EXISTS FOR (p:Policy) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT target_name IF NOT EXISTS FOR (t:Target) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT legislation_name IF NOT EXISTS FOR (l:Legislation) REQUIRE l.name IS UNIQUE;
CREATE CONSTRAINT country_iso IF NOT EXISTS FOR (c:Country) REQUIRE c.iso_code IS UNIQUE;
CREATE CONSTRAINT renewable_name IF NOT EXISTS FOR (r:RenewableSource) REQUIRE r.name IS UNIQUE;
CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE;

// Create indexes for performance
CREATE INDEX policy_year IF NOT EXISTS FOR (p:Policy) ON (p.year);
CREATE INDEX target_deadline IF NOT EXISTS FOR (t:Target) ON (t.deadline);
CREATE INDEX legislation_year IF NOT EXISTS FOR (l:Legislation) ON (l.year);
```
