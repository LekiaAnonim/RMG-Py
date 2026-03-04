"""
Local SQLite Database Manager for Import Voting System

Provides local storage for votes with no network dependency.
Can work completely offline and sync later with remote API.
"""
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os


CHEMKIN_REACTIONS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS chemkin_reactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        reaction_index INTEGER NOT NULL,
        reaction_string TEXT NOT NULL,
        reactant_labels TEXT,
        product_labels TEXT,
        kinetics_type TEXT,
        kinetics_comment TEXT,
        is_matched BOOLEAN DEFAULT 0,
        is_identified BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES import_jobs(job_id) ON DELETE CASCADE,
        UNIQUE(job_id, reaction_index)
    )
"""

CHEMKIN_REACTIONS_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS idx_chemkin_reactions_job_id 
    ON chemkin_reactions(job_id)
"""

CHEMKIN_REACTIONS_MATCHED_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS idx_chemkin_reactions_matched 
    ON chemkin_reactions(job_id, is_matched)
"""


class VoteLocalDB:
    """
    Local database manager for votes using SQLite
    Ensures data persistence without network dependency
    """
    
    def __init__(self, db_path: str = './import_votes.db'):
        """
        Initialize local database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        # Use a longer timeout and autocommit to reduce lock contention
        self.conn = sqlite3.connect(
            db_path,
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        self.conn.row_factory = sqlite3.Row  # Access columns by name

        self.conn.execute("PRAGMA journal_mode=DELETE")  # Avoid WAL mode on NFS
        self.conn.execute("PRAGMA synchronous=NORMAL")   # Less aggressive syncing
        self.conn.execute("PRAGMA busy_timeout=30000")   # Wait up to 30s if locked

        self._configure_connection()
        self.logger = logging.getLogger(__name__)
        self._create_tables()
        self._ensure_import_jobs_columns()
        self._ensure_identified_species_columns()  # NEW: Track processed species
        self.logger.info(f"Initialized local database: {db_path}")

    def _configure_connection(self):
        """Configure SQLite connection pragmas for better concurrency."""
        # Use DELETE mode (default) - simpler and avoids -shm/-wal files
        # that can cause "locking protocol" errors with remote SSH access
        self.conn.execute("PRAGMA journal_mode=DELETE;")
        # Reduce write lock duration and improve concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        # Enforce foreign key constraints
        self.conn.execute("PRAGMA foreign_keys=ON;")
        # Wait up to 30 seconds when the database is locked
        self.conn.execute("PRAGMA busy_timeout=30000;")
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Import jobs table - stores full progress data matching progress_json()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS import_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                model_name TEXT,
                species_file TEXT,
                reactions_file TEXT,
                thermo_file TEXT,
                status TEXT DEFAULT 'active',
                total_species INTEGER DEFAULT 0,
                identified_species INTEGER DEFAULT 0,
                confirmed_species INTEGER DEFAULT 0,
                processed_species INTEGER DEFAULT 0,
                unprocessed_species INTEGER DEFAULT 0,
                tentative_species INTEGER DEFAULT 0,
                unidentified_species INTEGER DEFAULT 0,
                total_reactions INTEGER DEFAULT 0,
                matched_reactions INTEGER DEFAULT 0,
                unmatched_reactions INTEGER DEFAULT 0,
                thermo_matches_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Species votes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS species_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                chemkin_label TEXT NOT NULL,
                chemkin_formula TEXT,
                rmg_species_label TEXT,
                rmg_species_smiles TEXT,
                rmg_species_index INTEGER,
                rmg_species_formula TEXT,
                vote_count INTEGER DEFAULT 0,
                enthalpy_discrepancy REAL,
                confidence_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES import_jobs(job_id) ON DELETE CASCADE,
                UNIQUE(job_id, chemkin_label, rmg_species_index)
            )
        """)
        
        # Voting reactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voting_reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_vote_id INTEGER NOT NULL,
                chemkin_reaction_str TEXT,
                edge_reaction_str TEXT,
                reaction_family TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (species_vote_id) REFERENCES species_votes(id) ON DELETE CASCADE
            )
        """)
        
        # Identified species table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS identified_species (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                chemkin_label TEXT NOT NULL,
                chemkin_formula TEXT,
                rmg_species_label TEXT,
                rmg_species_smiles TEXT,
                rmg_species_index INTEGER,
                identification_method TEXT DEFAULT 'auto',
                identified_by TEXT,
                enthalpy_discrepancy REAL,
                notes TEXT,
                identified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES import_jobs(job_id) ON DELETE CASCADE,
                UNIQUE(job_id, chemkin_label)
            )
        """)
        
        # Blocked matches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blocked_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                chemkin_label TEXT NOT NULL,
                rmg_species_label TEXT,
                rmg_species_smiles TEXT,
                rmg_species_index INTEGER,
                blocked_by TEXT,
                reason TEXT,
                blocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES import_jobs(job_id) ON DELETE CASCADE,
                UNIQUE(job_id, chemkin_label, rmg_species_index)
            )
        """)
        
        # Thermo matches table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thermo_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_vote_id INTEGER NOT NULL,
                library_name TEXT NOT NULL,
                library_species_name TEXT NOT NULL,
                name_matches BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (species_vote_id) REFERENCES species_votes(id) ON DELETE CASCADE,
                UNIQUE(species_vote_id, library_name, library_species_name)
            )
        """)
        
        # Sync log table (track what's been synced to remote)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                sync_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                record_count INTEGER,
                success BOOLEAN,
                error_message TEXT,
                synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES import_jobs(job_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_species_votes_job_id 
            ON species_votes(job_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_species_votes_chemkin_label 
            ON species_votes(chemkin_label)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_identified_species_job_id 
            ON identified_species(job_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_blocked_matches_job_id 
            ON blocked_matches(job_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_thermo_matches_species_vote_id 
            ON thermo_matches(species_vote_id)
        """)

        cursor.execute(CHEMKIN_REACTIONS_TABLE_SQL)
        cursor.execute(CHEMKIN_REACTIONS_INDEX_SQL)
        cursor.execute(CHEMKIN_REACTIONS_MATCHED_INDEX_SQL)
        
        self.conn.commit()

    def _ensure_import_jobs_columns(self):
        """Ensure import_jobs has all expected columns (schema upgrade for older DBs)."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(import_jobs)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "model_name": "TEXT",
            "species_file": "TEXT",
            "reactions_file": "TEXT",
            "thermo_file": "TEXT",
            "status": "TEXT DEFAULT 'active'",
            "total_species": "INTEGER DEFAULT 0",
            "identified_species": "INTEGER DEFAULT 0",
            "confirmed_species": "INTEGER DEFAULT 0",
            "processed_species": "INTEGER DEFAULT 0",
            "unprocessed_species": "INTEGER DEFAULT 0",
            "tentative_species": "INTEGER DEFAULT 0",
            "unidentified_species": "INTEGER DEFAULT 0",
            "total_reactions": "INTEGER DEFAULT 0",
            "matched_reactions": "INTEGER DEFAULT 0",
            "unmatched_reactions": "INTEGER DEFAULT 0",
            "thermo_matches_count": "INTEGER DEFAULT 0",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        }

        for column, col_type in expected_columns.items():
            if column not in existing_columns:
                cursor.execute(f"ALTER TABLE import_jobs ADD COLUMN {column} {col_type}")

        self.conn.commit()
    
    def _ensure_identified_species_columns(self):
        """
        Ensure identified_species has is_processed column for tracking 
        which species have been fully processed (limit_enlarge completed).
        
        This is critical for restart optimization - we skip expensive 
        limit_enlarge() calls for already-processed species.
        """
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(identified_species)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # New columns needed for restart optimization
        expected_columns = {
            "is_processed": "BOOLEAN DEFAULT 0",  # True after limit_enlarge() completes
            "processed_at": "TIMESTAMP",          # When processing completed
        }

        for column, col_type in expected_columns.items():
            if column not in existing_columns:
                self.logger.info(f"Adding column {column} to identified_species table")
                cursor.execute(f"ALTER TABLE identified_species ADD COLUMN {column} {col_type}")

        self.conn.commit()

    # ==================== Import Job Methods ====================
    
    def create_or_get_job(self, job_id: str, model_name: str = "",
                         species_file: str = "", reactions_file: str = "",
                         thermo_file: str = ""):
        """Create or retrieve an import job"""
        cursor = self.conn.cursor()
        
        # Try to get existing job
        cursor.execute("SELECT * FROM import_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        
        if row:
            self.logger.info(f"Retrieved existing job: {job_id}")
            return dict(row)
        
        # Create new job
        cursor.execute("""
            INSERT INTO import_jobs 
            (job_id, model_name, species_file, reactions_file, thermo_file)
            VALUES (?, ?, ?, ?, ?)
        """, (job_id, model_name, species_file, reactions_file, thermo_file))
        
        self.conn.commit()
        self.logger.info(f"Created new job: {job_id}")
        
        # Return the created job
        cursor.execute("SELECT * FROM import_jobs WHERE job_id = ?", (job_id,))
        return dict(cursor.fetchone())
    
    def update_job_statistics(self, job_id: str, total_species: int = None,
                             identified_species: int = None,
                             total_reactions: int = None,
                             matched_reactions: int = None,
                             confirmed_species: int = None,
                             processed_species: int = None,
                             unprocessed_species: int = None,
                             tentative_species: int = None,
                             unidentified_species: int = None,
                             unmatched_reactions: int = None,
                             thermo_matches_count: int = None):
        """Update job statistics with full progress data"""
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if total_species is not None:
            updates.append("total_species = ?")
            params.append(total_species)

        if identified_species is not None:
            updates.append("identified_species = ?")
            params.append(identified_species)
        
        if total_reactions is not None:
            updates.append("total_reactions = ?")
            params.append(total_reactions)
        
        if matched_reactions is not None:
            updates.append("matched_reactions = ?")
            params.append(matched_reactions)
        
        if confirmed_species is not None:
            updates.append("confirmed_species = ?")
            params.append(confirmed_species)
        
        if processed_species is not None:
            updates.append("processed_species = ?")
            params.append(processed_species)
        
        if unprocessed_species is not None:
            updates.append("unprocessed_species = ?")
            params.append(unprocessed_species)
        
        if tentative_species is not None:
            updates.append("tentative_species = ?")
            params.append(tentative_species)
        
        if unidentified_species is not None:
            updates.append("unidentified_species = ?")
            params.append(unidentified_species)
        
        if unmatched_reactions is not None:
            updates.append("unmatched_reactions = ?")
            params.append(unmatched_reactions)
        
        if thermo_matches_count is not None:
            updates.append("thermo_matches_count = ?")
            params.append(thermo_matches_count)
        
        if not updates:
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(job_id)
        
        query = f"UPDATE import_jobs SET {', '.join(updates)} WHERE job_id = ?"
        cursor.execute(query, params)
        self.conn.commit()
        
        return True
    
    def update_progress_from_importer(self, job_id: str, progress_data: dict):
        """
        Update job statistics from progress_json() data format
        
        Args:
            job_id: Import job identifier
            progress_data: Dictionary from progress_json() with keys:
                - processed, unprocessed, confirmed, tentative
                - unidentified, unconfirmed, total
                - unmatchedreactions, totalreactions, thermomatches
        """
        return self.update_job_statistics(
            job_id=job_id,
            total_species=progress_data.get('total'),
            identified_species=progress_data.get('confirmed'),
            confirmed_species=progress_data.get('confirmed'),
            processed_species=progress_data.get('processed'),
            unprocessed_species=progress_data.get('unprocessed'),
            tentative_species=progress_data.get('tentative'),
            unidentified_species=progress_data.get('unidentified'),
            total_reactions=progress_data.get('totalreactions'),
            unmatched_reactions=progress_data.get('unmatchedreactions'),
            matched_reactions=(progress_data.get('totalreactions', 0) - 
                             progress_data.get('unmatchedreactions', 0)),
            thermo_matches_count=progress_data.get('thermomatches'),
        )
    
    def update_job_status(self, job_id: str, status: str):
        """Update job status"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE import_jobs 
            SET status = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE job_id = ?
        """, (status, job_id))
        self.conn.commit()
        return True
    
    # ==================== Species Vote Methods ====================
    
    def save_votes(self, job_id: str, votes_dict: Dict, enthalpy_calculator=None):
        """
        Save votes to local database
        
        Args:
            job_id: Import job identifier
            votes_dict: Dictionary in format {chemkin_label: {rmg_species: set([(ck_rxn, rmg_rxn), ...])}}
            enthalpy_calculator: Optional callable(chemkin_label, rmg_species) -> float (enthalpy discrepancy in kJ/mol)
            
        Returns:
            Number of votes saved
        """
        cursor = self.conn.cursor()
        saved_count = 0
        
        for chemkin_label, possible_matches in votes_dict.items():
            for rmg_species, voting_reactions in possible_matches.items():
                # Extract species information safely
                rmg_species_label = getattr(rmg_species, 'label', str(rmg_species))
                rmg_species_index = getattr(rmg_species, 'index', -1)
                
                # Get SMILES safely
                rmg_species_smiles = ''
                try:
                    if hasattr(rmg_species, 'molecule') and rmg_species.molecule:
                        rmg_species_smiles = rmg_species.molecule[0].to_smiles()
                except:
                    pass
                
                # Get formula safely
                chemkin_formula = ''
                rmg_species_formula = ''
                try:
                    if hasattr(rmg_species, 'get_formula'):
                        rmg_species_formula = rmg_species.get_formula()
                        chemkin_formula = rmg_species_formula
                except:
                    pass
                
                # Calculate enthalpy discrepancy if calculator provided
                enthalpy_disc = None
                if enthalpy_calculator:
                    try:
                        enthalpy_disc = enthalpy_calculator(chemkin_label, rmg_species)
                    except Exception as e:
                        self.logger.debug(f"Could not calculate enthalpy for {chemkin_label}: {e}")
                
                # Insert or update species vote
                cursor.execute("""
                    INSERT OR REPLACE INTO species_votes 
                    (job_id, chemkin_label, chemkin_formula, rmg_species_label, 
                     rmg_species_smiles, rmg_species_index, rmg_species_formula,
                     vote_count, enthalpy_discrepancy, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    job_id, chemkin_label, chemkin_formula, rmg_species_label,
                    rmg_species_smiles, rmg_species_index, rmg_species_formula,
                    len(voting_reactions), enthalpy_disc
                ))
                
                species_vote_id = cursor.lastrowid
                
                # Delete old voting reactions for this vote
                cursor.execute(
                    "DELETE FROM voting_reactions WHERE species_vote_id = ?",
                    (species_vote_id,)
                )
                
                # Insert voting reactions
                for chemkin_reaction, edge_reaction in voting_reactions:
                    reaction_family = getattr(edge_reaction, 'family', '')
                    
                    cursor.execute("""
                        INSERT INTO voting_reactions 
                        (species_vote_id, chemkin_reaction_str, edge_reaction_str, reaction_family)
                        VALUES (?, ?, ?, ?)
                    """, (
                        species_vote_id,
                        str(chemkin_reaction),
                        str(edge_reaction),
                        reaction_family
                    ))
                
                saved_count += 1
        
        self.conn.commit()
        self.logger.info(f"Saved {saved_count} species votes to local database")
        return saved_count
    
    def load_votes(self, job_id: str):
        """Load all votes for a job"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT sv.*, 
                   COUNT(vr.id) as reaction_count
            FROM species_votes sv
            LEFT JOIN voting_reactions vr ON sv.id = vr.species_vote_id
            WHERE sv.job_id = ?
            GROUP BY sv.id
            ORDER BY sv.vote_count DESC, sv.chemkin_label
        """, (job_id,))
        
        votes = [dict(row) for row in cursor.fetchall()]
        self.logger.info(f"Loaded {len(votes)} votes for job {job_id}")
        return votes
    
    def load_votes_for_reconstruction(self, job_id: str):
        """
        Load votes with all data needed to reconstruct self.votes dictionary.
        
        Returns a list of dicts with structure:
        {
            'chemkin_label': str,
            'rmg_species_smiles': str,
            'rmg_species_index': int,
            'vote_count': int,
            'voting_reactions': [{'chemkin_reaction_str': str, 'edge_reaction_str': str}, ...]
        }
        
        This is optimized for reconstructing votes quickly on job restart.
        """
        cursor = self.conn.cursor()
        
        # Get all votes for this job
        cursor.execute("""
            SELECT id, chemkin_label, rmg_species_smiles, rmg_species_index, vote_count
            FROM species_votes
            WHERE job_id = ?
            ORDER BY chemkin_label, vote_count DESC
        """, (job_id,))
        
        votes_data = []
        for row in cursor.fetchall():
            vote_id = row['id']
            vote_dict = {
                'chemkin_label': row['chemkin_label'],
                'rmg_species_smiles': row['rmg_species_smiles'],
                'rmg_species_index': row['rmg_species_index'],
                'vote_count': row['vote_count'],
                'voting_reactions': []
            }
            
            # Get voting reactions for this vote
            cursor.execute("""
                SELECT chemkin_reaction_str, edge_reaction_str, reaction_family
                FROM voting_reactions
                WHERE species_vote_id = ?
            """, (vote_id,))
            
            vote_dict['voting_reactions'] = [
                {
                    'chemkin_reaction_str': r['chemkin_reaction_str'],
                    'edge_reaction_str': r['edge_reaction_str'],
                    'reaction_family': r['reaction_family'],
                }
                for r in cursor.fetchall()
            ]
            
            votes_data.append(vote_dict)
        
        self.logger.info(f"Loaded {len(votes_data)} vote records for reconstruction (job {job_id})")
        return votes_data
    
    def load_voting_reactions(self, species_vote_id: int):
        """Load voting reactions for a specific species vote"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM voting_reactions 
            WHERE species_vote_id = ?
            ORDER BY created_at
        """, (species_vote_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def clear_votes(self, job_id: str, chemkin_labels: List[str] = None,
                   rmg_species_indices: List[int] = None):
        """Clear votes for specific species"""
        cursor = self.conn.cursor()
        
        query = "DELETE FROM species_votes WHERE job_id = ?"
        params = [job_id]
        
        conditions = []
        if chemkin_labels:
            placeholders = ','.join('?' * len(chemkin_labels))
            conditions.append(f"chemkin_label IN ({placeholders})")
            params.extend(chemkin_labels)
        
        if rmg_species_indices:
            placeholders = ','.join('?' * len(rmg_species_indices))
            conditions.append(f"rmg_species_index IN ({placeholders})")
            params.extend(rmg_species_indices)
        
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"
        
        cursor.execute(query, params)
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        self.logger.info(f"Cleared {deleted_count} votes from local database")
        return deleted_count
    
    def backfill_enthalpy(self, job_id: str, enthalpy_calculator, species_dict: Dict = None):
        """
        Backfill enthalpy_discrepancy for existing votes that have NULL values.
        
        Args:
            job_id: Import job identifier
            enthalpy_calculator: Callable(chemkin_label, rmg_species) -> float
            species_dict: Optional dict mapping rmg_species_index -> rmg_species object
                         If not provided, will only update based on available info
            
        Returns:
            Number of votes updated
        """
        cursor = self.conn.cursor()
        
        # Get all votes with NULL enthalpy
        cursor.execute("""
            SELECT id, chemkin_label, rmg_species_index, rmg_species_smiles
            FROM species_votes
            WHERE job_id = ? AND enthalpy_discrepancy IS NULL
        """, (job_id,))
        
        rows = cursor.fetchall()
        updated_count = 0
        
        for row in rows:
            vote_id = row['id']
            chemkin_label = row['chemkin_label']
            rmg_index = row['rmg_species_index']
            smiles = row['rmg_species_smiles']
            
            # Try to get the RMG species object
            rmg_species = None
            if species_dict and rmg_index in species_dict:
                rmg_species = species_dict[rmg_index]
            
            if rmg_species is None:
                continue
                
            # Calculate enthalpy
            try:
                enthalpy_disc = enthalpy_calculator(chemkin_label, rmg_species)
                if enthalpy_disc is not None:
                    cursor.execute("""
                        UPDATE species_votes 
                        SET enthalpy_discrepancy = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (enthalpy_disc, vote_id))
                    updated_count += 1
            except Exception as e:
                self.logger.debug(f"Could not calculate enthalpy for {chemkin_label}: {e}")
        
        self.conn.commit()
        self.logger.info(f"Backfilled enthalpy for {updated_count} votes")
        return updated_count
    
    # ==================== Thermo Match Methods ====================
    
    def save_thermo_matches(self, job_id: str, thermo_matches_dict: Dict):
        """
        Save thermo matches to local database
        
        Args:
            job_id: Import job identifier
            thermo_matches_dict: Dictionary in format {chemkin_label: {rmg_species: [(lib_name, lib_species_name), ...]}}
            
        Returns:
            Number of thermo matches saved
        """
        cursor = self.conn.cursor()
        saved_count = 0
        
        for chemkin_label, rmg_species_dict in thermo_matches_dict.items():
            for rmg_species, library_matches in rmg_species_dict.items():
                # Get the species_vote_id for this chemkin_label and rmg_species
                rmg_species_index = getattr(rmg_species, 'index', -1)
                
                cursor.execute("""
                    SELECT id FROM species_votes 
                    WHERE job_id = ? AND chemkin_label = ? AND rmg_species_index = ?
                """, (job_id, chemkin_label, rmg_species_index))
                
                row = cursor.fetchone()
                if not row:
                    # No species vote found - skip this thermo match
                    self.logger.warning(
                        f"No species_vote found for {chemkin_label} -> RMG#{rmg_species_index}, "
                        f"skipping thermo matches"
                    )
                    continue
                
                species_vote_id = row[0]
                
                # Delete old thermo matches for this species vote
                cursor.execute(
                    "DELETE FROM thermo_matches WHERE species_vote_id = ?",
                    (species_vote_id,)
                )
                
                # Insert thermo matches
                for library_name, library_species_name in library_matches:
                    # Check if library species name matches chemkin label
                    name_matches = (library_species_name.lower() == chemkin_label.lower())
                    
                    cursor.execute("""
                        INSERT INTO thermo_matches 
                        (species_vote_id, library_name, library_species_name, name_matches)
                        VALUES (?, ?, ?, ?)
                    """, (
                        species_vote_id,
                        library_name,
                        library_species_name,
                        1 if name_matches else 0
                    ))
                    
                    saved_count += 1
        
        self.conn.commit()
        self.logger.info(f"Saved {saved_count} thermo matches to local database")
        return saved_count
    
    def load_thermo_matches(self, species_vote_id: int):
        """Load thermo matches for a specific species vote"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM thermo_matches 
            WHERE species_vote_id = ?
            ORDER BY name_matches DESC, library_name
        """, (species_vote_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def load_all_thermo_matches(self, job_id: str):
        """
        Load all thermo matches for a job, organized by chemkin_label and rmg_species
        
        Returns:
            Dictionary with structure matching votes loading format for easy sync
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                sv.chemkin_label,
                sv.rmg_species_index,
                sv.rmg_species_label,
                sv.rmg_species_smiles,
                tm.library_name,
                tm.library_species_name,
                tm.name_matches
            FROM species_votes sv
            JOIN thermo_matches tm ON sv.id = tm.species_vote_id
            WHERE sv.job_id = ?
            ORDER BY sv.chemkin_label, sv.rmg_species_index, tm.name_matches DESC
        """, (job_id,))
        
        # Organize by chemkin_label -> rmg_species_index -> list of matches
        thermo_data = {}
        for row in cursor.fetchall():
            chemkin_label = row['chemkin_label']
            rmg_index = row['rmg_species_index']
            
            if chemkin_label not in thermo_data:
                thermo_data[chemkin_label] = {}
            
            if rmg_index not in thermo_data[chemkin_label]:
                thermo_data[chemkin_label][rmg_index] = {
                    'rmg_species_label': row['rmg_species_label'],
                    'rmg_species_smiles': row['rmg_species_smiles'],
                    'thermo_matches': []
                }
            
            thermo_data[chemkin_label][rmg_index]['thermo_matches'].append({
                'library': row['library_name'],
                'species_name': row['library_species_name'],
                'name_matches': bool(row['name_matches'])
            })
        
        return thermo_data
    
    def clear_thermo_matches(self, job_id: str, chemkin_label: str = None):
        """Clear thermo matches for a job or specific species"""
        cursor = self.conn.cursor()
        
        if chemkin_label:
            cursor.execute("""
                DELETE FROM thermo_matches 
                WHERE species_vote_id IN (
                    SELECT id FROM species_votes 
                    WHERE job_id = ? AND chemkin_label = ?
                )
            """, (job_id, chemkin_label))
        else:
            cursor.execute("""
                DELETE FROM thermo_matches 
                WHERE species_vote_id IN (
                    SELECT id FROM species_votes WHERE job_id = ?
                )
            """, (job_id,))
        
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        self.logger.info(f"Cleared {deleted_count} thermo matches from local database")
        return deleted_count
    
    # ==================== Identified Species Methods ====================
    
    def save_identified_species(self, job_id: str, chemkin_label: str,
                               rmg_species, identified_by: str = None,
                               identification_method: str = 'auto',
                               enthalpy_discrepancy: float = None,
                               notes: str = ""):
        """
        Save an identified species, preserving is_processed status if already set.
        
        IMPORTANT: This method preserves the is_processed and processed_at columns
        when updating an existing record. This is critical for restart optimization -
        if a species was already processed (limit_enlarge completed), we must not
        reset that flag when the importer restarts and re-saves the identified species.
        """
        cursor = self.conn.cursor()
        
        # Extract species information safely
        rmg_species_label = getattr(rmg_species, 'label', str(rmg_species))
        rmg_species_index = getattr(rmg_species, 'index', None)
        
        # Get SMILES safely
        rmg_species_smiles = ''
        try:
            if hasattr(rmg_species, 'molecule') and rmg_species.molecule:
                rmg_species_smiles = rmg_species.molecule[0].to_smiles()
        except:
            pass
        
        # Get formula safely
        chemkin_formula = ''
        try:
            if hasattr(rmg_species, 'get_formula'):
                chemkin_formula = rmg_species.get_formula()
        except:
            pass
        
        # Check if record exists and get current is_processed status
        cursor.execute("""
            SELECT is_processed, processed_at 
            FROM identified_species 
            WHERE job_id = ? AND chemkin_label = ?
        """, (job_id, chemkin_label))
        existing = cursor.fetchone()
        
        if existing:
            # UPDATE existing record, preserving is_processed and processed_at
            cursor.execute("""
                UPDATE identified_species 
                SET chemkin_formula = ?, rmg_species_label = ?, 
                    rmg_species_smiles = ?, rmg_species_index = ?, 
                    identification_method = ?, identified_by = ?, 
                    enthalpy_discrepancy = ?, notes = ?
                WHERE job_id = ? AND chemkin_label = ?
            """, (
                chemkin_formula, rmg_species_label,
                rmg_species_smiles, rmg_species_index, identification_method,
                identified_by, enthalpy_discrepancy, notes,
                job_id, chemkin_label
            ))
            self.logger.debug(f"Updated identified species (preserved is_processed={existing['is_processed']}): {chemkin_label}")
        else:
            # INSERT new record
            cursor.execute("""
                INSERT INTO identified_species 
                (job_id, chemkin_label, chemkin_formula, rmg_species_label, 
                 rmg_species_smiles, rmg_species_index, identification_method,
                 identified_by, enthalpy_discrepancy, notes, is_processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                job_id, chemkin_label, chemkin_formula, rmg_species_label,
                rmg_species_smiles, rmg_species_index, identification_method,
                identified_by, enthalpy_discrepancy, notes
            ))
            self.logger.debug(f"Inserted new identified species: {chemkin_label}")
        
        self.conn.commit()
        self.logger.info(f"Saved identified species: {chemkin_label}")
        return True
    
    def load_identified_species(self, job_id: str):
        """Load list of identified species labels"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT chemkin_label 
            FROM identified_species 
            WHERE job_id = ?
            ORDER BY identified_at
        """, (job_id,))
        
        labels = [row['chemkin_label'] for row in cursor.fetchall()]
        self.logger.info(f"Loaded {len(labels)} identified species for job {job_id}")
        return labels
    
    def load_identified_species_full(self, job_id: str):
        """Load full identified species data"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM identified_species 
            WHERE job_id = ?
            ORDER BY identified_at
        """, (job_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def mark_species_processed(self, job_id: str, chemkin_label: str):
        """
        Mark a species as fully processed (limit_enlarge completed).
        
        This is called AFTER limit_enlarge() successfully completes for a species.
        On restart, species marked as processed will be skipped, avoiding
        the expensive edge reaction generation.
        
        Args:
            job_id: The import job ID
            chemkin_label: The chemkin species label that was processed
            
        Returns:
            True if successful, False otherwise
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE identified_species 
            SET is_processed = 1, processed_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND chemkin_label = ?
        """, (job_id, chemkin_label))
        
        self.conn.commit()
        
        if cursor.rowcount > 0:
            self.logger.debug(f"Marked species as processed: {chemkin_label}")
            return True
        else:
            self.logger.warning(f"Species not found for marking processed: {chemkin_label}")
            return False
    
    def load_processed_species(self, job_id: str) -> List[str]:
        """
        Load list of species labels that have been fully processed.
        
        These species had limit_enlarge() completed and should be SKIPPED
        on restart to avoid expensive re-computation.
        
        Args:
            job_id: The import job ID
            
        Returns:
            List of chemkin labels that are already processed
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT chemkin_label 
            FROM identified_species 
            WHERE job_id = ? AND is_processed = 1
            ORDER BY processed_at
        """, (job_id,))
        
        labels = [row['chemkin_label'] for row in cursor.fetchall()]
        self.logger.info(f"Loaded {len(labels)} already-processed species for job {job_id}")
        return labels
    
    def get_processing_status(self, job_id: str) -> Dict:
        """
        Get summary of processing status for a job.
        
        Returns:
            Dict with counts of identified, processed, and unprocessed species
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_identified,
                SUM(CASE WHEN is_processed = 1 THEN 1 ELSE 0 END) as total_processed,
                SUM(CASE WHEN is_processed = 0 OR is_processed IS NULL THEN 1 ELSE 0 END) as total_unprocessed
            FROM identified_species 
            WHERE job_id = ?
        """, (job_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'total_identified': row['total_identified'] or 0,
                'total_processed': row['total_processed'] or 0,
                'total_unprocessed': row['total_unprocessed'] or 0
            }
        return {'total_identified': 0, 'total_processed': 0, 'total_unprocessed': 0}

    # ==================== Blocked Match Methods ====================
    
    def save_blocked_match(self, job_id: str, chemkin_label: str,
                          rmg_species, blocked_by: str = None,
                          reason: str = ""):
        """Save a blocked match"""
        cursor = self.conn.cursor()
        
        # Extract species information safely
        rmg_species_label = getattr(rmg_species, 'label', str(rmg_species))
        rmg_species_index = getattr(rmg_species, 'index', None)
        
        # Get SMILES safely
        rmg_species_smiles = ''
        try:
            if hasattr(rmg_species, 'molecule') and rmg_species.molecule:
                rmg_species_smiles = rmg_species.molecule[0].to_smiles()
        except:
            pass
        
        cursor.execute("""
            INSERT OR REPLACE INTO blocked_matches 
            (job_id, chemkin_label, rmg_species_label, rmg_species_smiles,
             rmg_species_index, blocked_by, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, chemkin_label, rmg_species_label, rmg_species_smiles,
            rmg_species_index, blocked_by, reason
        ))
        
        self.conn.commit()
        self.logger.info(f"Saved blocked match: {chemkin_label}")
        return True
    
    def load_blocked_matches(self, job_id: str):
        """Load all blocked matches for a job"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM blocked_matches 
            WHERE job_id = ?
            ORDER BY blocked_at
        """, (job_id,))
        
        blocked = [dict(row) for row in cursor.fetchall()]
        self.logger.info(f"Loaded {len(blocked)} blocked matches for job {job_id}")
        return blocked
    


    def save_chemkin_reactions(self, job_id: str, reactions: list, 
                               unmatched_reactions: list = None):
        """
        Save chemkin reactions to the database.
        
        Args:
            job_id: Import job identifier
            reactions: List of chemkin reaction objects
            unmatched_reactions: List of reactions that haven't been matched yet
                                (used to determine is_matched status)
        
        Returns:
            Number of reactions saved
        """
        cursor = self.conn.cursor()
        saved_count = 0
        
        # Create set of unmatched reaction indices for quick lookup
        unmatched_set = set()
        if unmatched_reactions:
            for rxn in unmatched_reactions:
                try:
                    idx = reactions.index(rxn)
                    unmatched_set.add(idx)
                except ValueError:
                    pass
        
        for idx, reaction in enumerate(reactions):
            try:
                # Extract reaction information
                reaction_string = str(reaction)
                
                # Get reactant and product labels
                reactant_labels = ','.join([
                    getattr(s, 'label', str(s)) for s in reaction.reactants
                ])
                product_labels = ','.join([
                    getattr(s, 'label', str(s)) for s in reaction.products
                ])
                
                # Get kinetics information
                kinetics_type = ''
                kinetics_comment = ''
                if hasattr(reaction, 'kinetics') and reaction.kinetics:
                    kinetics_type = type(reaction.kinetics).__name__
                    kinetics_comment = getattr(reaction.kinetics, 'comment', '')[:500]  # Limit length
                
                # Determine match status
                is_matched = idx not in unmatched_set
                
                # Check if all species are identified (have molecules)
                is_identified = True
                for species in list(reaction.reactants) + list(reaction.products):
                    if not hasattr(species, 'molecule') or not species.molecule:
                        is_identified = False
                        break
                
                # Insert or update
                cursor.execute("""
                    INSERT INTO chemkin_reactions 
                    (job_id, reaction_index, reaction_string, reactant_labels, 
                     product_labels, kinetics_type, kinetics_comment, 
                     is_matched, is_identified, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(job_id, reaction_index) DO UPDATE SET
                        reaction_string = excluded.reaction_string,
                        reactant_labels = excluded.reactant_labels,
                        product_labels = excluded.product_labels,
                        kinetics_type = excluded.kinetics_type,
                        kinetics_comment = excluded.kinetics_comment,
                        is_matched = excluded.is_matched,
                        is_identified = excluded.is_identified,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    job_id, idx, reaction_string, reactant_labels,
                    product_labels, kinetics_type, kinetics_comment,
                    1 if is_matched else 0, 1 if is_identified else 0
                ))
                
                saved_count += 1
                
            except Exception as e:
                self.logger.warning(f"Could not save reaction {idx}: {e}")
        
        self.conn.commit()
        self.logger.info(f"Saved {saved_count} chemkin reactions to database")
        return saved_count
    
    def load_chemkin_reactions(self, job_id: str, matched_only: bool = False,
                               identified_only: bool = False,
                               limit: int = None, offset: int = 0):
        """
        Load chemkin reactions from database.
        
        Args:
            job_id: Import job identifier
            matched_only: If True, only return matched reactions
            identified_only: If True, only return fully identified reactions
            limit: Maximum number of reactions to return
            offset: Number of reactions to skip (for pagination)
        
        Returns:
            List of reaction dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM chemkin_reactions WHERE job_id = ?"
        params = [job_id]
        
        if matched_only:
            query += " AND is_matched = 1"
        
        if identified_only:
            query += " AND is_identified = 1"
        
        query += " ORDER BY reaction_index"
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        reactions = [dict(row) for row in cursor.fetchall()]
        return reactions
    
    def update_reaction_match_status(self, job_id: str, reaction_index: int,
                                     is_matched: bool = None,
                                     is_identified: bool = None):
        """
        Update the match/identified status of a specific reaction.
        
        Args:
            job_id: Import job identifier
            reaction_index: Index of the reaction to update
            is_matched: New matched status (or None to leave unchanged)
            is_identified: New identified status (or None to leave unchanged)
        
        Returns:
            True if update was successful
        """
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if is_matched is not None:
            updates.append("is_matched = ?")
            params.append(1 if is_matched else 0)
        
        if is_identified is not None:
            updates.append("is_identified = ?")
            params.append(1 if is_identified else 0)
        
        if not updates:
            return False
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.extend([job_id, reaction_index])
        
        query = f"""
            UPDATE chemkin_reactions 
            SET {', '.join(updates)}
            WHERE job_id = ? AND reaction_index = ?
        """
        
        cursor.execute(query, params)
        self.conn.commit()
        
        return cursor.rowcount > 0
    
    def get_reaction_statistics(self, job_id: str):
        """
        Get reaction statistics for a job.
        
        Returns:
            Dictionary with reaction counts
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_matched = 1 THEN 1 ELSE 0 END) as matched,
                SUM(CASE WHEN is_identified = 1 THEN 1 ELSE 0 END) as identified,
                SUM(CASE WHEN is_matched = 0 THEN 1 ELSE 0 END) as unmatched
            FROM chemkin_reactions
            WHERE job_id = ?
        """, (job_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'total': row['total'] or 0,
                'matched': row['matched'] or 0,
                'identified': row['identified'] or 0,
                'unmatched': row['unmatched'] or 0
            }
        return {'total': 0, 'matched': 0, 'identified': 0, 'unmatched': 0}
    
    def bulk_update_reaction_status(self, job_id: str, 
                                    matched_indices: set = None,
                                    identified_indices: set = None):
        """
        Bulk update reaction statuses.
        
        Args:
            job_id: Import job identifier
            matched_indices: Set of reaction indices that are matched
            identified_indices: Set of reaction indices that are fully identified
        
        Returns:
            Number of reactions updated
        """
        cursor = self.conn.cursor()
        updated_count = 0
        
        # Reset all to unmatched/unidentified first
        cursor.execute("""
            UPDATE chemkin_reactions 
            SET is_matched = 0, is_identified = 0, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, (job_id,))
        
        # Update matched reactions
        if matched_indices:
            placeholders = ','.join('?' * len(matched_indices))
            cursor.execute(f"""
                UPDATE chemkin_reactions 
                SET is_matched = 1, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND reaction_index IN ({placeholders})
            """, [job_id] + list(matched_indices))
            updated_count += cursor.rowcount
        
        # Update identified reactions
        if identified_indices:
            placeholders = ','.join('?' * len(identified_indices))
            cursor.execute(f"""
                UPDATE chemkin_reactions 
                SET is_identified = 1, updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND reaction_index IN ({placeholders})
            """, [job_id] + list(identified_indices))
            updated_count += cursor.rowcount
        
        self.conn.commit()
        return updated_count
    
    # ==================== Sync Methods ====================
    
    def log_sync(self, job_id: str, sync_type: str, direction: str,
                record_count: int, success: bool, error_message: str = ""):
        """Log a sync operation"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO sync_log 
            (job_id, sync_type, direction, record_count, success, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (job_id, sync_type, direction, record_count, success, error_message))
        
        self.conn.commit()
        return True
    
    def get_last_sync(self, job_id: str, sync_type: str = None):
        """Get the last sync operation for a job"""
        cursor = self.conn.cursor()
        
        if sync_type:
            cursor.execute("""
                SELECT * FROM sync_log 
                WHERE job_id = ? AND sync_type = ?
                ORDER BY synced_at DESC
                LIMIT 1
            """, (job_id, sync_type))
        else:
            cursor.execute("""
                SELECT * FROM sync_log 
                WHERE job_id = ?
                ORDER BY synced_at DESC
                LIMIT 1
            """, (job_id,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    # ==================== Export/Import Methods ====================
    
    def export_to_json(self, job_id: str, output_file: str):
        """Export all data for a job to JSON file"""
        try:
            data = {
                'job': self.create_or_get_job(job_id),
                'votes': self.load_votes(job_id),
                'identified': self.load_identified_species_full(job_id),
                'blocked': self.load_blocked_matches(job_id),
                'exported_at': datetime.now().isoformat()
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Exported job {job_id} to {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False
    
    def import_from_json(self, input_file: str):
        """Import data from JSON file"""
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            job_id = data['job']['job_id']
            
            # Create job
            self.create_or_get_job(
                job_id=job_id,
                model_name=data['job'].get('model_name', '')
            )
            
            # Import identified species (preserving is_processed if exists)
            for species in data.get('identified', []):
                cursor = self.conn.cursor()
                
                # Check if record exists
                cursor.execute("""
                    SELECT is_processed, processed_at 
                    FROM identified_species 
                    WHERE job_id = ? AND chemkin_label = ?
                """, (job_id, species['chemkin_label']))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing, preserve is_processed
                    cursor.execute("""
                        UPDATE identified_species 
                        SET chemkin_formula = ?, rmg_species_label = ?,
                            rmg_species_smiles = ?, rmg_species_index = ?, 
                            identification_method = ?, identified_by = ?, 
                            enthalpy_discrepancy = ?, notes = ?
                        WHERE job_id = ? AND chemkin_label = ?
                    """, (
                        species.get('chemkin_formula', ''), species['rmg_species_label'],
                        species.get('rmg_species_smiles', ''), species.get('rmg_species_index'),
                        species.get('identification_method', 'auto'), species.get('identified_by'),
                        species.get('enthalpy_discrepancy'), species.get('notes', ''),
                        job_id, species['chemkin_label']
                    ))
                else:
                    # Insert new record with is_processed from import data or default 0
                    is_processed = species.get('is_processed', 0)
                    processed_at = species.get('processed_at')
                    cursor.execute("""
                        INSERT INTO identified_species 
                        (job_id, chemkin_label, chemkin_formula, rmg_species_label,
                         rmg_species_smiles, rmg_species_index, identification_method,
                         identified_by, enthalpy_discrepancy, notes, is_processed, processed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        job_id, species['chemkin_label'], species.get('chemkin_formula', ''),
                        species['rmg_species_label'], species.get('rmg_species_smiles', ''),
                        species.get('rmg_species_index'), species.get('identification_method', 'auto'),
                        species.get('identified_by'), species.get('enthalpy_discrepancy'),
                        species.get('notes', ''), is_processed, processed_at
                    ))
            
            self.conn.commit()
            self.logger.info(f"Imported job {job_id} from {input_file}")
            return job_id
        
        except Exception as e:
            self.logger.error(f"Import failed: {e}")
            return None
    
    # ==================== Utility Methods ====================
    
    def get_statistics(self, job_id: str):
        """Get statistics for a job"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Get job info
        cursor.execute("SELECT * FROM import_jobs WHERE job_id = ?", (job_id,))
        job = cursor.fetchone()
        if job:
            stats['job'] = dict(job)
        
        # Count votes
        cursor.execute(
            "SELECT COUNT(*) as count FROM species_votes WHERE job_id = ?",
            (job_id,)
        )
        stats['total_votes'] = cursor.fetchone()['count']
        
        # Count unique species with votes
        cursor.execute(
            "SELECT COUNT(DISTINCT chemkin_label) as count FROM species_votes WHERE job_id = ?",
            (job_id,)
        )
        stats['species_with_votes'] = cursor.fetchone()['count']
        
        # Count identified species
        cursor.execute(
            "SELECT COUNT(*) as count FROM identified_species WHERE job_id = ?",
            (job_id,)
        )
        stats['identified_species'] = cursor.fetchone()['count']
        
        # Count blocked matches
        cursor.execute(
            "SELECT COUNT(*) as count FROM blocked_matches WHERE job_id = ?",
            (job_id,)
        )
        stats['blocked_matches'] = cursor.fetchone()['count']
        
        return stats
    
    def vacuum(self):
        """Optimize database (reclaim space)"""
        self.conn.execute("VACUUM")
        self.logger.info("Database vacuumed")
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        self.logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
